
import argparse
import ast
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
import numpy as np

from src.prompts.method_a import STARTING_PROGRAM
from src.prompts.method_b import generate_prompt

from src.llm_client import (
    call_llm,
    credit_remaining,
    _price,
    MODEL,
    PRICE_MODEL,
    MAX_TOKENS,
)
from src.generate_instances import load_dataset
from src.sandbox import run_candidate


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "pilot"
BEST_DIR = OUTPUT_DIR / "best_candidates"

NUM_CALLS = 300
TIMEOUT = 10.0
BUDGET_USD = 15.0

#Method B settings
FUNCTIONS_PER_PROMPT = 2   # k = 2 parents per prompt
SELECTION_TEMPERATURE = 0.1
RANDOM_SEED = 0

# Method A's prompt, and Method B's 1st prompt - Best Fit as priority_fn_v0.
INITIAL_PROMPT = generate_prompt([STARTING_PROGRAM])

def can_make_request(prompt, budget_state):
    """Check whether another API request can safely fit within the budget."""

    if budget_state["stop"]:
        return False

    try:
        # Remaining credit on the OpenRouter API key.
        account_remaining = credit_remaining()

        # Published input and output prices, in USD per token.
        input_price, output_price = _price(PRICE_MODEL)

        if account_remaining is None:
            raise ValueError("OpenRouter remaining credit is unknown.")

        # Use UTF-8 bytes as a conservative upper bound for prompt tokens.
        estimated_input_tokens = int(len(prompt.encode("utf-8")) * 1.25) + 1

        # Reserve enough for the maximum possible response.
        estimated_request_cost = (
            estimated_input_tokens * input_price
            + MAX_TOKENS * output_price
        )

    except Exception as e:
        budget_state["stop"] = True
        budget_state["reason"] = f"Could not verify budget: {e}"
        print(f"\nSTOPPING PILOT: {budget_state['reason']}")
        return False

    experiment_remaining = BUDGET_USD - budget_state["spent"]

    available = min(experiment_remaining, float(account_remaining))

    if estimated_request_cost > available:
        budget_state["stop"] = True
        budget_state["reason"] = (
            f"Next request could cost up to ${estimated_request_cost:.4f}, "
            f"but only ${available:.4f} is available."
        )
        print(f"\nSTOPPING PILOT: {budget_state['reason']}")
        return False

    return True


def update_budget_after_call(api_result, budget_state):
    """Add the actual API cost to the experiment's spending total."""

    cost = api_result.get("cost")

    if cost is None:
        budget_state["stop"] = True
        budget_state["reason"] = (
            "API request cost is unknown. Stopping to avoid overspending."
        )
        print(f"\nSTOPPING PILOT: {budget_state['reason']}")
        return

    budget_state["spent"] += float(cost)

    print(
        f"Experiment spending: "
        f"${budget_state['spent']:.4f} / ${BUDGET_USD:.2f}"
    )

    if budget_state["spent"] >= BUDGET_USD:
        budget_state["stop"] = True
        budget_state["reason"] = "Experiment budget reached."


# ============================================================
# Local demo candidate
# ============================================================

DEMO_SOURCE = """
def priority_fn(item_size, open_bin_remaining_capacities, capacity):
    gaps = open_bin_remaining_capacities - item_size
    scores = -gaps
    return np.append(scores, -capacity - 1)
""".strip()


# ============================================================
# Utility functions
# ============================================================

def save_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


def extract_candidate(response, version):
    """Extract priority_fn_v{version} from the LLM response and rename it
    back to priority_fn for the sandbox to run"""
    if not response:
        return None, "empty response"

    text = response.strip()

    # Remove Markdown code fences if the model includes them.
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Start from the requested function and rename it to priority_fn.
    match = re.search(rf"(?m)^def\s+priority_fn_v{version}\s*\(", text)
    if match is None:
        return None, f"response does not define priority_fn_v{version}"
    text = re.sub(
        rf"\bpriority_fn_v{version}\b", "priority_fn", text[match.start():]
    )

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return None, f"Python syntax error: {exc}"

    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "priority_fn"
    ]

    if len(functions) != 1:
        return None, "response must contain exactly one priority_fn"

    return text.strip(), None


def format_value(value):
    if value is None:
        return "Not available"
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def select_parents(valid_candidates, rng):
    """Parent selection for Method B.

    valid_candidates: list of {"id", "code", "score"} (score = % excess).
    Candidates with the same score are counted once, keeping the shortest code.
    Parents are drawn with replacement (follows funsearch), 
    with probability proportional to exp(-score/T) (better candidates have higher chance of getting selected), 
    then sorted from worse to better.
    """
    by_score = {}
    for c in valid_candidates:
        kept = by_score.get(c["score"])
        if kept is None or len(c["code"]) < len(kept["code"]):
            by_score[c["score"]] = c
    unique = list(by_score.values())

    scores = np.array([c["score"] for c in unique])
    weights = np.exp(-(scores - scores.min()) / SELECTION_TEMPERATURE)
    k = min(FUNCTIONS_PER_PROMPT, len(unique))
    idx = rng.choice(len(unique), size=k, p=weights / weights.sum())

    parents = [unique[i] for i in idx]
    return sorted(parents, key=lambda c: c["score"], reverse=True)


# ============================================================
# Candidate evaluation and logging
# ============================================================

def evaluate_and_record(method, generation, response, val_data,
                        run_id, api_result, parents, cumulative_tokens):
    """parents: list of {"id", "code", "score"} shown in the prompt."""
    code, extraction_error = extract_candidate(response, len(parents))

    record = {
        "method": method,
        "run_id": run_id,
        "generation": generation,
        "parent_ids": [p["id"] for p in parents],
        "parent_scores": [p["score"] for p in parents],
        "num_parents": len(parents),
        "cumulative_tokens": cumulative_tokens,
        "api_error": api_result.get("error"),
        "api_cost": api_result.get("cost"),
        "prompt_tokens": (api_result.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (api_result.get("usage") or {}).get("completion_tokens"),
        "total_tokens": api_result.get("total_tokens"),
        "api_latency_s": api_result.get("latency_s"),
        "candidate_code": code,
        "extraction_error": extraction_error,
    }

    if code is None:
        record["evaluation"] = {
            "status": "invalid",
            "error": extraction_error,
        }
        record["valid"] = False
    else:
        try:
            result = run_candidate(
                source=code,
                dataset=val_data,
                timeout=TIMEOUT,
            )
        except Exception as e:
            result = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }

        record["evaluation"] = result
        record["valid"] = result.get("status") == "ok"

    save_jsonl(OUTPUT_DIR / f"candidates_{run_id}.jsonl", record)

    return record


def record_starting_program(method, val_data, run_id):
    """Evaluate Best Fit and log it as generation 0."""
    result = run_candidate(
        source=STARTING_PROGRAM, dataset=val_data, timeout=TIMEOUT
    )
    if result.get("status") != "ok":
        raise RuntimeError(f"Starting program failed: {result}")
    record = {
        "method": method,
        "run_id": run_id,
        "generation": 0,
        "parent_ids": [],
        "parent_scores": [],
        "num_parents": 0,
        "cumulative_tokens": 0,
        "total_tokens": 0,
        "api_cost": 0.0,
        "api_error": None,
        "candidate_code": STARTING_PROGRAM,
        "extraction_error": None,
        "evaluation": result,
        "valid": True,
    }
    save_jsonl(OUTPUT_DIR / f"candidates_{run_id}.jsonl", record)
    return record


# ============================================================
# Method A: Independent sampling
# ============================================================

def run_method_a(val_data, run_id, budget_state):
    """Every call uses the same prompt: Best Fit as priority_fn_v0."""
    print("\nStarting Method A: Independent Sampling")

    start = record_starting_program("A", val_data, run_id)
    starting_parent = {
        "id": 0, "code": STARTING_PROGRAM, "score": start["evaluation"]["score"]
    }
    best_score = starting_parent["score"]
    cumulative_tokens = 0
    records = [start]

    for generation in range(1, NUM_CALLS + 1):
        print(f"\nMethod A - Call {generation}/{NUM_CALLS}")

        if not can_make_request(INITIAL_PROMPT, budget_state):
            break

        api_result = call_llm(
            prompt=INITIAL_PROMPT,
            run_id=run_id,
            method="A",
            generation=generation,
        )
        update_budget_after_call(api_result, budget_state)
        cumulative_tokens += api_result.get("total_tokens") or 0

        record = evaluate_and_record(
            "A", generation, api_result.get("text"), val_data, run_id,
            api_result, [starting_parent], cumulative_tokens,
        )
        records.append(record)

        score = record["evaluation"].get("score")
        if record["valid"] and score < best_score:
            best_score = score
            BEST_DIR.mkdir(parents=True, exist_ok=True)
            (BEST_DIR / f"{run_id}.py").write_text(
                record["candidate_code"], encoding="utf-8"
            )

        print(
            f"Status: {record['evaluation'].get('status')} | "
            f"Score: {format_value(score)} | Best: {format_value(best_score)}"
        )
        if budget_state["stop"]:
            break

    return records


# ============================================================
# Method B: LLM-guided evolution
# ============================================================

def run_method_b(val_data, run_id, budget_state):
    """Each call: choose parents from the valid candidates, ask for the next
    version, and add the child to the valid candidates."""
    print("\nStarting Method B: LLM-Guided Evolution")

    rng = np.random.default_rng(RANDOM_SEED)

    start = record_starting_program("B", val_data, run_id)
    valid_candidates = [{
        "id": 0, "code": STARTING_PROGRAM, "score": start["evaluation"]["score"]
    }]
    best_score = valid_candidates[0]["score"]
    cumulative_tokens = 0
    records = [start]

    for generation in range(1, NUM_CALLS + 1):
        print(f"\nMethod B - Generation {generation}/{NUM_CALLS}")

        parents = select_parents(valid_candidates, rng)
        prompt = generate_prompt([p["code"] for p in parents])

        if not can_make_request(prompt, budget_state):
            break

        api_result = call_llm(
            prompt=prompt,
            run_id=run_id,
            method="B",
            generation=generation,
        )
        update_budget_after_call(api_result, budget_state)
        cumulative_tokens += api_result.get("total_tokens") or 0

        record = evaluate_and_record(
            "B", generation, api_result.get("text"), val_data, run_id,
            api_result, parents, cumulative_tokens,
        )
        records.append(record)

        score = record["evaluation"].get("score")
        if record["valid"]:
            valid_candidates.append({
                "id": generation,
                "code": record["candidate_code"],
                "score": score,
            })
            if score < best_score:
                best_score = score
                BEST_DIR.mkdir(parents=True, exist_ok=True)
                (BEST_DIR / f"{run_id}.py").write_text(
                    record["candidate_code"], encoding="utf-8"
                )

        print(
            f"Parents: {record['parent_ids']} | "
            f"Status: {record['evaluation'].get('status')} | "
            f"Score: {format_value(score)} | Best: {format_value(best_score)}"
        )
        if budget_state["stop"]:
            break

    return records


# ============================================================
# Summary
# ============================================================

def make_summary(records):
    # Generation 0 is the starting program (Best Fit), not an API call.
    start = records[0] if records and records[0]["generation"] == 0 else None
    records = [r for r in records if r["generation"] != 0]

    successful = [
        r for r in records
        if r.get("api_error") is None
    ]

    valid = [
        r for r in records
        if r.get("valid")
    ]

    scores = [
        r["evaluation"]["score"]
        for r in valid
    ]

    costs = [
        r["api_cost"]
        for r in records
        if isinstance(r.get("api_cost"), (int, float))
    ]

    tokens = [
        r["total_tokens"]
        for r in records
        if isinstance(r.get("total_tokens"), (int, float))
    ]

    return {
        "starting_program_score": (
            start["evaluation"]["score"] if start else None
        ),
        "api_calls": len(records),
        "successful_api_calls": len(successful),
        "valid_candidates": len(valid),
        "invalid_or_timeout_candidates": len(records) - len(valid),
        "best_score": min(scores) if scores else None,
        "mean_score": statistics.mean(scores) if scores else None,
        "total_api_cost_usd": sum(costs),
        "mean_api_cost_usd": statistics.mean(costs) if costs else None,
        "total_tokens": sum(tokens),
        "mean_tokens": statistics.mean(tokens) if tokens else None,
    }


# ============================================================
# Local test
# ============================================================

def local_test():
    print("Running local sandbox test. No API calls will be made.")

    val_data = load_dataset("val")

    for name, source in (("demo", DEMO_SOURCE),
                         ("starting program", STARTING_PROGRAM)):
        result = run_candidate(
            source=source,
            dataset=val_data,
            timeout=TIMEOUT,
        )
        print(name, result.get("status"), result.get("score"))
        if result.get("status") != "ok":
            raise RuntimeError(f"Local {name} test failed.")

    print("\nLocal test passed!")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-test",
        action="store_true",
        help="Test the sandbox using a local candidate without API calls.",
    )
    parser.add_argument(
        "--method",
        choices=["A", "B", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.local_test:
        local_test()
        return

    val_data = load_dataset("val")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_records = {}

    budget_state = {
        "spent": 0.0,
        "stop": False,
        "reason": None,
    }

    if args.method in ("A", "all"):
        run_id_a = f"pilot_A_{timestamp}"
        all_records["A"] = run_method_a(
            val_data, run_id_a, budget_state
        )

    if args.method in ("B", "all"):
        run_id_b = f"pilot_B_{timestamp}"
        all_records["B"] = run_method_b(
            val_data, run_id_b, budget_state
        )

    summary = {
        method: make_summary(records)
        for method, records in all_records.items()
    }

    summary["budget"] = {
        "experiment_budget_usd": BUDGET_USD,
        "actual_spending_usd": round(budget_state["spent"], 6),
        "stopped_early": budget_state["stop"],
        "stop_reason": budget_state["reason"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / f"summary_{timestamp}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n========== PILOT SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
