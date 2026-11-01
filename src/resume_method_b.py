
import argparse
import json
from datetime import datetime
from pathlib import Path

from src.llm_client import call_llm
from src.sandbox import run_candidate, _load_dataset
from src.run_pilot import (
    NUM_CALLS,
    TIMEOUT,
    BUDGET_USD,
    OUTPUT_DIR,
    BEST_DIR,
    INITIAL_PROMPT,
    can_make_request,
    update_budget_after_call,
    extract_candidate,
    build_evolution_prompt,
    format_value,
    make_summary,
)

ROOT = Path(__file__).resolve().parent.parent

BACKUP_DIR = ROOT / "backups" / "pilot_20261101"
HISTORY_PATH = BACKUP_DIR / "candidates_B.jsonl"
ORIGINAL_SUMMARY_PATH = BACKUP_DIR / "summary_20261101_102125.json"

# Recorded spending from the original experiment:
# Method A: $0.05342625
# Method B: $0.06368475
PREVIOUS_SPENDING = 0.117111


def load_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


def evaluate_candidate(response, generation, train_data, api_result):
    code, extraction_error = extract_candidate(response)

    record = {
        "method": "B",
        "generation": generation,
        "api_error": api_result.get("error"),
        "api_cost": api_result.get("cost"),
        "prompt_tokens": (api_result.get("usage") or {}).get(
            "prompt_tokens"
        ),
        "completion_tokens": (api_result.get("usage") or {}).get(
            "completion_tokens"
        ),
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

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-test",
        action="store_true",
        help="Check the saved history without making API calls.",
    )
    args = parser.parse_args()

    # Load the original experiment history.
    history = load_jsonl(HISTORY_PATH)

    valid_history = [
        r for r in history
        if r.get("valid")
        and r.get("candidate_code")
        and r.get("evaluation", {}).get("status") == "ok"
    ]

    if not valid_history:
        raise RuntimeError("No valid historical Method B candidates found.")

    # Restore the latest valid candidate as the evolution parent.
    parent_record = max(
        valid_history,
        key=lambda r: r["generation"],
    )
    parent_code = parent_record["candidate_code"]
    parent_result = parent_record["evaluation"]

    # Restore the best candidate across all historical generations.
    best_record = min(
        valid_history,
        key=lambda r: r["evaluation"]["score"],
    )
    best_code = best_record["candidate_code"]
    best_score = best_record["evaluation"]["score"]

    valid_count = len(valid_history)

    # Generation 34 failed, so retry it.
    next_generation = max(
        r["generation"] for r in valid_history
    ) + 1

    print("========== RESUME CHECK ==========")
    print(f"Historical candidate records: {len(history)}")
    print(f"Historical valid candidates: {valid_count}")
    print(f"Restored parent generation: {parent_record['generation']}")
    print(f"Parent score: {parent_result['score']:.6f}")
    print(f"Best historical generation: {best_record['generation']}")
    print(f"Best historical score: {best_score:.6f}")
    print(f"Next generation to attempt: {next_generation}")
    print(f"Previous experiment spending: ${PREVIOUS_SPENDING:.6f}")

    if args.local_test:
        print("\nLocal test passed. No API calls were made.")
        return

    # Create unique output names so original files are untouched.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"pilot_B_resume_{timestamp}"

    candidate_path = (
        OUTPUT_DIR / f"candidates_B_resume_{timestamp}.jsonl"
    )
    summary_path = (
        OUTPUT_DIR / f"summary_B_resume_{timestamp}.json"
    )
    best_path = (
        BEST_DIR / f"method_B_resume_{timestamp}.py"
    )
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    best_path.write_text(best_code, encoding="utf-8")

    train_data = _load_dataset("train")

    # Include all previous spending in the $15 budget.
    budget_state = {
        "spent": PREVIOUS_SPENDING,
        "stop": False,
        "reason": None,
    }

    new_records = []

    print("\nStarting resumed Method B.")
    print(f"Target: {NUM_CALLS} valid candidates in total.")

    while next_generation <= NUM_CALLS:
        generation = next_generation

        print(
            f"\nMethod B - Generation {generation} "
            f"({valid_count}/{NUM_CALLS} valid candidates)"
        )

        prompt_text = build_evolution_prompt(
            parent_code=parent_code,
            parent_result=parent_result,
            best_score=best_score,
            candidate_count=len(history) + len(new_records),
        )

        if not can_make_request(prompt_text, budget_state):
            break

        api_result = call_llm(
            prompt=prompt_text,
            run_id=run_id,
            method="B",
            generation=generation,
        )

        update_budget_after_call(api_result, budget_state)

        record = evaluate_candidate(
            api_result.get("text"),
            generation,
            train_data,
            api_result,
        )

        # Save every attempt, including failed attempts.
        save_jsonl(candidate_path, record)
        new_records.append(record)

        result = record["evaluation"]
        score = result.get("score")

        if record["valid"]:
            valid_count += 1

            # Every valid candidate becomes the next parent.
            parent_code = record["candidate_code"]
            parent_result = result

            # Update the best candidate only if its score improves.
            if score < best_score:
                best_score = score
                best_code = record["candidate_code"]

            # Save the best candidate found so far.
            best_path.write_text(best_code, encoding="utf-8")

        print(
            f"Status: {result.get('status')} | "
            f"Score: {format_value(score)}"
        )

        if budget_state["stop"]:
            break

        next_generation += 1

    # Combine the historical and resumed records for reporting.
    combined_records = history + new_records

    original_summary = json.loads(
        ORIGINAL_SUMMARY_PATH.read_text(encoding="utf-8")
    )

    summary = {
        "A_original": original_summary["A"],
        "B_combined": make_summary(combined_records),
        "B_resume_only": make_summary(new_records),
        "resume_metadata": {
            "historical_valid_candidates": len(valid_history),
            "new_valid_candidates": sum(
                1 for r in new_records if r.get("valid")
            ),
            "total_valid_candidates": valid_count,
            "target_valid_candidates": NUM_CALLS,
            "last_historical_parent_generation": (
                parent_record["generation"]
            ),
            "first_generation_attempted": (
                new_records[0]["generation"] if new_records else None
            ),
        },
        "budget": {
            "experiment_budget_usd": BUDGET_USD,
            "previous_spending_usd": PREVIOUS_SPENDING,
            "new_spending_usd": round(
                budget_state["spent"] - PREVIOUS_SPENDING, 6
            ),
            "total_spending_usd": round(budget_state["spent"], 6),
            "stopped_early": budget_state["stop"],
            "stop_reason": budget_state["reason"],
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n========== RESUME SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nNew candidates: {candidate_path}")
    print(f"New API logs: {OUTPUT_DIR.parent / 'llm_logs' / (run_id + '.jsonl')}")
    print(f"New summary: {summary_path}")
    print(f"Best candidate: {best_path}")


if __name__ == "__main__":
    main()

