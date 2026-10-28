
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

for split in ["train", "val","test"]:

    df = pd.read_csv(
        OUTPUT_DIR / f"baseline_comparison_{split}.csv"
    )

    summary = df.groupby("family").agg(
        num_instances=("instance_id", "count"),

        pct_any_diff=(
            "max_gap",
            lambda x: 100 * (x > 0).mean()
        ),

        pct_ff_bf_diff=(
            "ff_bf_diff",
            lambda x: 100 * (x != 0).mean()
        ),

        pct_bf_ab_diff=(
            "bf_ab_diff",
            lambda x: 100 * (x != 0).mean()
        ),

        mean_ff_bf_gap=(
            "ff_bf_diff",
            lambda x: x.abs().mean()
        ),

        mean_bf_ab_gap=(
            "bf_ab_diff",
            lambda x: x.abs().mean()
        ),

        mean_max_gap=("max_gap", "mean"),
        largest_gap=("max_gap", "max")
    ).round(2)

    print(f"\n===== {split.upper()} =====")
    print(summary.to_string())

    summary.to_csv(
        OUTPUT_DIR / f"baseline_summary_{split}.csv"
    )

df["bf_improvement_pct"] = (
    100 * (df["FF"] - df["BF"]) / df["FF"]
)

df["ab_improvement_pct"] = (
    100 * (df["BF"] - df["AB"]) / df["BF"]
)

print(
    df.groupby("family")[
        ["bf_improvement_pct", "ab_improvement_pct"]
    ].mean().round(3)
)    
