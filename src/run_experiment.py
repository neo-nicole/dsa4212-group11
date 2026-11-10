import argparse
import json
import math
import builtins
from datetime import datetime
from pathlib import Path

try:
    from . import run_pilot as pilot
except ImportError:
    import run_pilot as pilot

from src.generate_instances import load_dataset


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str), encoding="utf-8")


def install_token_limit(max_tokens):
    """Enforce an independent token allowance for each method/repetition.

    The pilot search loops call these wrappers. A token stop is local to the
    current run; only the USD/credit/API-safety guard can stop all runs.
    """
    original_check = pilot.can_make_request
    original_update = pilot.update_budget_after_call
    state = {"used": 0, "limit": max_tokens, "stop_reason": None, "calls": 0}

    def check(prompt, budget):
        if budget["stop"] or state["stop_reason"]:
            return False
        #estimate of prompt tokens + maximum response tokens.
        reserved = int(len(prompt.encode("utf-8")) * 1.25) + 1 + pilot.MAX_TOKENS
        if state["used"] + reserved > state["limit"]:
            state["stop_reason"] = "Insufficient remaining tokens for next request"
            print(f"Stopping this method/repetition: {state['stop_reason']}. "
                  f"Used {state['used']:,}/{state['limit']:,} tokens.")
            return False
        return original_check(prompt, budget)

    def update(result, budget):
        original_update(result, budget)
        usage = result.get("total_tokens")
        if isinstance(usage, bool) or not isinstance(usage, (int, float)) or not math.isfinite(usage) or usage < 0:
            budget["stop"] = True
            budget["reason"] = "API token usage unavailable or invalid; cannot enforce token limit."
            return
        state["used"] += int(usage)
        state["calls"] += 1
        print(
            f"Tokens: {state['used']:,} / {state['limit']:,} "
            f"(remaining: {state['limit'] - state['used']:,}) | "
            f"API calls: {state['calls']} | "
            f"Experiment cost: ${budget['spent']:.4f} / ${pilot.BUDGET_USD:.2f}"
        )
        if state["used"] >= state["limit"]:
            state["stop_reason"] = "Token limit reached"

    pilot.can_make_request = check
    pilot.update_budget_after_call = update
    return state


def reset_run_token_state(state):
    state["used"] = 0
    state["calls"] = 0
    state["stop_reason"] = None


def main():
    parser = argparse.ArgumentParser(description="Repeated main experiment for Methods A and B")
    parser.add_argument("--method", choices=("A", "B", "all"), default="all")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1_000_000, help="Independent cumulative token ceiling for each method in each repetition")
    parser.add_argument("--budget", type=float, default=12.0, help="Total USD cap across all methods and repetitions (separate from llm_client account limit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-temperature", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--prefix", default="final", help="Filename prefix expected by src.final_eval (default: final)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final_experiment"), help="Folder containing candidate logs for src.final_eval")
    args = parser.parse_args()
    if args.runs < 1 or args.max_tokens < 1 or args.budget <= 0 or args.timeout <= 0:
        parser.error("runs, max-tokens, budget and timeout must be positive")
    if not math.isfinite(args.budget) or not math.isfinite(args.timeout):
        parser.error("budget and timeout must be finite")
    if not math.isfinite(args.selection_temperature) or args.selection_temperature <= 0:
        parser.error("selection temperature must be finite and positive")

    if not args.prefix or not args.prefix.replace("-", "").isalnum():
        parser.error("--prefix must contain only letters, digits or hyphens")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.out_dir if args.out_dir.is_absolute() else pilot.ROOT / args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    pilot.OUTPUT_DIR = output
    pilot.BEST_DIR = output / "best_candidates"
    pilot.NUM_CALLS = args.max_tokens

    def experiment_print(*values, **kwargs):
        if values and isinstance(values[0], str):
            message = values[0]
            if message.startswith("\nMethod A - Call "):
                message = message.split("/", 1)[0]
            elif message.startswith("\nMethod B - Generation "):
                message = message.split("/", 1)[0]
            values = (message, *values[1:])
        builtins.print(*values, **kwargs)

    pilot.print = experiment_print
    token_state = install_token_limit(args.max_tokens)
    pilot.BUDGET_USD = args.budget
    pilot.TIMEOUT = args.timeout
    pilot.SELECTION_TEMPERATURE = args.selection_temperature

    train_data = load_dataset("train")
    methods = ["A", "B"] if args.method == "all" else [args.method]
    budget = {"spent": 0.0, "stop": False, "reason": None}
    rows = []
    config = vars(args).copy()
    config.update({"search_dataset": "train", "selection_and_testing": "src.final_eval", "model": pilot.MODEL, "request_max_output_tokens": pilot.MAX_TOKENS, "token_budget_scope": "per_method_per_repetition"})
    save_json(output / f"{args.prefix}_config_{stamp}.json", config)

    for repetition in range(args.runs):
        for method in methods:
            if budget["stop"]:
                break
            reset_run_token_state(token_state)
            pilot.RANDOM_SEED = args.seed + repetition
            seed = args.seed + repetition
            label = (f"{method}_T{args.selection_temperature:g}" if method == "B" else method)
            run_id = f"{args.prefix}_{label}_s{seed}_{stamp}"
            print(f"\n===== {run_id} =====")
            records = (pilot.run_method_a if method == "A" else pilot.run_method_b)(train_data, run_id, budget)
            summary = pilot.make_summary(records)
            summary["run_tokens"] = token_state["used"]
            summary["run_api_calls"] = token_state["calls"]
            summary["token_limit"] = args.max_tokens
            summary["token_stop_reason"] = token_state["stop_reason"]
            row = {
                "method": method, "repetition": repetition + 1,
                "seed": seed, "run_id": run_id,
                **summary,
            }
            rows.append(row)
            save_json(output / f"{args.prefix}_run_summaries_{stamp}.json", rows)
        if budget["stop"]:
            break

    aggregate = {}
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        aggregate[method] = {
            "completed_runs": len(subset),
            "total_api_calls": sum(r["api_calls"] for r in subset),
            "total_tokens": sum(r["run_tokens"] for r in subset),
            "total_api_cost_usd": sum(r["total_api_cost_usd"] for r in subset),
        }
    report = {"configuration": config, "budget": budget, "aggregate": aggregate, "runs": rows}
    save_json(output / f"{args.prefix}_experiment_summary_{stamp}.json", report)
    print("\n===== EXPERIMENT SUMMARY =====")
    print(json.dumps({"budget": budget, "aggregate": aggregate}, indent=2))
    print(f"\nCandidate logs saved to {output}")
    print(f"After all runs, execute: python -m src.final_eval --runs-dir {output} --prefix {args.prefix}")


if __name__ == "__main__":
    main()
