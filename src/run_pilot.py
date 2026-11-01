
import argparse
import ast
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from src.prompts.method_a import INITIAL_PROMPT
from src.prompts.method_b import EVOLUTION_PROMPT

from prompt_toolkit import prompt

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

NUM_CALLS = 50
TIMEOUT = 10.0
BUDGET_USD = 15.0

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


def extract_candidate(response):
    """Extract and check the Python function from the LLM response."""
    if not response:
        return None, "empty response"

    text = response.strip()

    # Remove Markdown code fences if the model includes them.
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # If there is explanatory text, start from the function definition.
    match = re.search(r"(?m)^def\s+priority_fn\s*\(", text)
    if match:
        text = text[match.start():]

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


def build_evolution_prompt(parent_code, parent_result,
                           best_score, candidate_count):
    family_scores = parent_result.get("family_scores") or {}

    return EVOLUTION_PROMPT.format(
        parent_code=parent_code,
        parent_score=format_value(parent_result.get("score")),
        mean_bins=format_value(parent_result.get("mean_bins")),
        uniform_200_score=format_value(
            family_scores.get("uniform_200")
        ),
        uniform_500_score=format_value(
            family_scores.get("uniform_500")
        ),
        runtime=format_value(parent_result.get("total_runtime")),
        status=parent_result.get("status", "unknown"),
        best_score=format_value(best_score),
        candidate_count=candidate_count,
    )


# ============================================================
# Candidate evaluation and logging
# ============================================================

def evaluate_and_record(method, generation, response, train_data,
                        run_id, api_result):
    code, extraction_error = extract_candidate(response)

    record = {
        "method": method,
        "generation": generation,
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
                dataset=train_data,
                timeout=TIMEOUT,
            )
        except Exception as e:
            result = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }

        record["evaluation"] = result
        record["valid"] = result.get("status") == "ok"

    save_jsonl(
        OUTPUT_DIR / f"candidates_{method}.jsonl",
        record,
    )

    return record


# ============================================================
# Method A: Independent sampling
# ============================================================

def run_method_a(train_data, run_id, budget_state):
    print("\nStarting Method A: Independent Sampling")

    best_code = None
    best_score = None
    records = []

    for generation in range(1, NUM_CALLS + 1):
        print(f"\nMethod A - Call {generation}/{NUM_CALLS}")

        if not can_make_request(INITIAL_PROMPT, budget_state):
            break

        api_result = call_llm(
            prompt=INITIAL_PROMPT,
            run_id=run_id,
            method="A",
            generation=None,
        )

        update_budget_after_call(api_result, budget_state)

        record = evaluate_and_record(
            "A", generation, api_result.get("text"),
            train_data, run_id, api_result,
        )
        records.append(record)

        result = record["evaluation"]
        score = result.get("score")

        if record["valid"] and (
            best_score is None or score < best_score
        ):
            best_score = score
            best_code = record["candidate_code"]

            BEST_DIR.mkdir(parents=True, exist_ok=True)
            (BEST_DIR / "method_A.py").write_text(
                best_code, encoding="utf-8"
            )

        print(
            f"Status: {result.get('status')} | "
            f"Score: {format_value(score)}"
        )
        if budget_state["stop"]:
            break

    return records


# ============================================================
# Method B: Evolutionary feedback
# ============================================================

def run_method_b(train_data, run_id, budget_state):
    print("\nStarting Method B: Evolutionary Feedback")

    parent_code = None
    parent_result = None

    best_code = None
    best_score = None
    records = []

    for generation in range(1, NUM_CALLS + 1):
        print(f"\nMethod B - Generation {generation}/{NUM_CALLS}")

        # First generation uses the exact same initial prompt as A.
        if generation == 1 or parent_code is None:
            prompt = INITIAL_PROMPT
        else:
            prompt = build_evolution_prompt(
                parent_code=parent_code,
                parent_result=parent_result,
                best_score=best_score,
                candidate_count=len(records),
            )

        # Check the budget before every generation, including generation 1.
        if not can_make_request(prompt, budget_state):
            break

        api_result = call_llm(
            prompt=prompt,
            run_id=run_id,
            method="B",
            generation=generation,
        )

        update_budget_after_call(api_result, budget_state)


        record = evaluate_and_record(
            "B", generation, api_result.get("text"),
            train_data, run_id, api_result,
        )
        records.append(record)

        result = record["evaluation"]
        score = result.get("score")

        # Every valid candidate becomes the next parent.
        # Invalid candidates and timeouts do not replace the parent.
        if record["valid"]:
            parent_code = record["candidate_code"]
            parent_result = result

            # Keep the best candidate separately.
            if best_score is None or score < best_score:
                best_score = score
                best_code = record["candidate_code"]

                BEST_DIR.mkdir(parents=True, exist_ok=True)
                (BEST_DIR / "method_B.py").write_text(
                    best_code, encoding="utf-8"
                )

        print(
            f"Status: {result.get('status')} | "
            f"Score: {format_value(score)}"
        )

        if budget_state["stop"]:
            break
    return records


# ============================================================
# Summary
# ============================================================

def make_summary(records):
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

    train_data = load_dataset("train")

    result = run_candidate(
        source=DEMO_SOURCE,
        dataset=train_data,
        timeout=TIMEOUT,
    )

    print(json.dumps(result, indent=2))

    if result.get("status") != "ok":
        raise RuntimeError("Local candidate test failed.")

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

    train_data = load_dataset("train")

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
            train_data, run_id_a, budget_state
        )

    if args.method in ("B", "all"):
        run_id_b = f"pilot_B_{timestamp}"
        all_records["B"] = run_method_b(
            train_data, run_id_b, budget_state
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
