import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from . import run_pilot as pilot
except ImportError:
    import run_pilot as pilot

from src.generate_instances import load_dataset
from src.sandbox import run_candidate


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def evaluate_code(code, dataset, timeout):
    result = run_candidate(source=code, dataset=dataset, timeout=timeout)
    if result.get("status") != "ok":
        return {"status": result.get("status", "error"), "error": result.get("error")}
    pilot.get_signature(result, dataset)  # Reject incomplete / malformed results.
    return {"status": "ok", "score": result["score"]}


def choose_on_validation(records, val_data, timeout):
    """Select one candidate per run without looking at test performance."""
    eligible = [r for r in records if r.get("valid") and r.get("candidate_code")]
    evaluated = []
    for record in eligible:
        try:
            result = evaluate_code(record["candidate_code"], val_data, timeout)
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
        evaluated.append({
            "generation": record["generation"],
            "train_score": record["evaluation"]["score"],
            "validation": result,
        })
    passing = [r for r in evaluated if r["validation"]["status"] == "ok"]
    if not passing:
        raise RuntimeError("No candidate passed validation, including Best Fit")
    winner = min(passing, key=lambda r: (r["validation"]["score"], r["train_score"], r["generation"]))
    chosen = next(r for r in eligible if r["generation"] == winner["generation"])
    return chosen, evaluated, winner


def install_token_limit(max_tokens):
    """Apply a per-run token ceiling to the existing pilot search loops.

    Actual token usage is read from the API response. Before each request we
    conservatively reserve the maximum output plus an input-token estimate.
    """
    original_check = pilot.can_make_request
    original_update = pilot.update_budget_after_call
    state = {"used": 0, "limit": max_tokens}

    def check(prompt, budget):
        if budget["stop"]:
            return False
        # Match the conservative input estimate used in run_pilot.py.
        reserved = int(len(prompt.encode("utf-8")) * 1.25) + 1 + pilot.MAX_TOKENS
        if state["used"] + reserved > state["limit"]:
            budget["stop"] = True
            budget["reason"] = "Per-run token limit reached (insufficient reserve for next request)."
            print(f"Stopping run: {budget['reason']} Used {state['used']:,}/{state['limit']:,} tokens.")
            return False
        return original_check(prompt, budget)

    def update(result, budget):
        original_update(result, budget)
        usage = result.get("total_tokens")
        if usage is None:
            budget["stop"] = True
            budget["reason"] = "API token usage unavailable; cannot enforce token limit."
            return
        state["used"] += int(usage)
        if state["used"] >= state["limit"]:
            budget["stop"] = True
            budget["reason"] = "Per-run token limit reached."

    pilot.can_make_request = check
    pilot.update_budget_after_call = update
    return state


def reset_run_token_state(state, budget):
    state["used"] = 0
    if budget["stop"] and budget["reason"] and budget["reason"].startswith("Per-run token limit"):
        budget["stop"] = False
        budget["reason"] = None


def main():
    parser = argparse.ArgumentParser(description="Repeated main experiment for Methods A and B")
    parser.add_argument("--method", choices=("A", "B", "all"), default="all")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1_000_000, help="Maximum cumulative prompt + completion tokens per method per run")
    parser.add_argument("--budget", type=float, default=150.0, help="Total USD cap across all runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--skip-test", action="store_true", help="Keep test untouched until final evaluation")
    args = parser.parse_args()
    if args.runs < 1 or args.max_tokens < 1 or args.budget <= 0 or args.timeout <= 0:
        parser.error("runs, max-tokens, budget and timeout must be positive")
    if not math.isfinite(args.budget) or not math.isfinite(args.timeout):
        parser.error("budget and timeout must be finite")
    if not math.isfinite(args.selection_temperature) or args.selection_temperature <= 0:
        parser.error("selection temperature must be finite and positive")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = pilot.ROOT / "outputs" / "experiment" / stamp
    output.mkdir(parents=True, exist_ok=False)
    pilot.OUTPUT_DIR = output
    pilot.BEST_DIR = output / "best_candidates"
    # Pilot loops require an iteration bound; the token guard, not this bound,
    # is the practical stopping criterion. Each request consumes >=1 token.
    pilot.NUM_CALLS = args.max_tokens
    token_state = install_token_limit(args.max_tokens)
    pilot.BUDGET_USD = args.budget
    pilot.TIMEOUT = args.timeout
    pilot.SELECTION_TEMPERATURE = args.selection_temperature

    train_data = load_dataset("train")
    val_data = load_dataset("val")
    methods = ["A", "B"] if args.method == "all" else [args.method]
    budget = {"spent": 0.0, "stop": False, "reason": None}
    rows = []
    selected = []
    config = vars(args).copy()
    config.update({"search_dataset": "train", "selection_dataset": "val", "test_dataset": None if args.skip_test else "test", "model": pilot.MODEL})
    save_json(output / "config.json", config)

    for repetition in range(args.runs):
        for method in methods:
            if budget["stop"]:
                break
            reset_run_token_state(token_state, budget)
            # Method B uses a separate, reproducible RNG stream per repetition.
            pilot.RANDOM_SEED = args.seed + repetition
            run_id = f"{method}_rep{repetition + 1:02d}_{stamp}"
            print(f"\n===== {run_id} =====")
            records = (pilot.run_method_a if method == "A" else pilot.run_method_b)(train_data, run_id, budget)
            summary = pilot.make_summary(records)
            summary["run_tokens"] = token_state["used"]
            summary["token_limit"] = args.max_tokens
            chosen, validation_results, winner = choose_on_validation(records, val_data, args.timeout)
            selected_path = output / "selected" / f"{run_id}.py"
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            selected_path.write_text(chosen["candidate_code"], encoding="utf-8")
            save_json(output / f"validation_{run_id}.json", validation_results)
            row = {
                "method": method, "repetition": repetition + 1,
                "seed": args.seed + repetition, "run_id": run_id,
                **summary,
                "selected_generation": winner["generation"],
                "selected_train_score": winner["train_score"],
                "selected_validation_score": winner["validation"]["score"],
                "selected_program": str(selected_path.relative_to(output)),
            }
            rows.append(row)
            selected.append((row, chosen["candidate_code"]))
            save_json(output / "run_summaries.json", rows)
            # A token-limit stop applies only to this run, not later runs.
            if budget["reason"] and budget["reason"].startswith("Per-run token limit"):
                budget["stop"] = False
                budget["reason"] = None
        if budget["stop"]:
            break

    # Only evaluate the already-frozen selections on test; never reselect using test.
    if not args.skip_test:
        test_data = load_dataset("test")
        for row, code in selected:
            try:
                row["test_evaluation"] = evaluate_code(code, test_data, args.timeout)
            except Exception as exc:
                row["test_evaluation"] = {"status": "error", "error": str(exc)}

    aggregate = {}
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        test_scores = [r["test_evaluation"]["score"] for r in subset if r.get("test_evaluation", {}).get("status") == "ok"]
        aggregate[method] = {
            "completed_runs": len(subset),
            "mean_validation_score": float(np.mean([r["selected_validation_score"] for r in subset])) if subset else None,
            "mean_test_score": float(np.mean(test_scores)) if test_scores else None,
            "std_test_score": float(np.std(test_scores, ddof=1)) if len(test_scores) > 1 else None,
            "total_api_calls": sum(r["api_calls"] for r in subset),
            "total_tokens": sum(r["run_tokens"] for r in subset),
            "total_api_cost_usd": sum(r["total_api_cost_usd"] for r in subset),
        }
    report = {"configuration": config, "budget": budget, "aggregate": aggregate, "runs": rows}
    save_json(output / "final_summary.json", report)
    print("\n===== EXPERIMENT SUMMARY =====")
    print(json.dumps({"budget": budget, "aggregate": aggregate}, indent=2))
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
