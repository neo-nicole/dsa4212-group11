
import json
from datetime import datetime

from src.run_pilot import (
    BUDGET_USD,
    OUTPUT_DIR,
    make_summary,
    run_method_b,
)
from src.sandbox import _load_dataset


def main():
    train_data = _load_dataset("train")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"pilot_B_{timestamp}"

    budget_state = {
        "spent": 0.0,
        "stop": False,
        "reason": None,
    }

    records = run_method_b(train_data, run_id, budget_state)

    summary = {
        "B": make_summary(records),
        "budget": {
            "experiment_budget_usd": BUDGET_USD,
            "actual_spending_usd": round(budget_state["spent"], 6),
            "stopped_early": budget_state["stop"],
            "stop_reason": budget_state["reason"],
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / f"summary_B_{timestamp}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n========== METHOD B SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
