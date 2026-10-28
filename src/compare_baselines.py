
import json
from pathlib import Path
from collections import defaultdict
import statistics
import csv

from baselines import (
    simulate,
    first_fit,
    best_fit,
    make_ab_first_fit
)

ROOT = Path(__file__).resolve().parent.parent
#Input datasets
DATA_DIR = ROOT / "data"
#Output comparison results
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AB_A = 20
AB_B = 23

heuristics = {
    "FF": first_fit,
    "BF": best_fit,
    "AB": make_ab_first_fit(AB_A, AB_B)
}


def compare_baselines(dataset):
    rows = []

    for instance_id, instance in dataset.items():
        items = instance["items"]
        capacity = instance["capacity"]
        family = instance["family"]

        bins = {
            name: simulate(items, capacity, h)
            for name, h in heuristics.items()
        }

        rows.append({
            "instance_id": instance_id,
            "family": family,
            "num_items": len(items),
            "FF": bins["FF"],
            "BF": bins["BF"],
            "AB": bins["AB"],
            "ff_bf_diff": bins["FF"] - bins["BF"],
            "bf_ab_diff": bins["BF"] - bins["AB"],
            "ff_ab_diff": bins["FF"] - bins["AB"],
            "max_gap": max(bins.values()) - min(bins.values())
        })

    return rows


def print_summary(rows, split):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["family"]].append(row)

    print(f"\n========== {split.upper()} ==========")

    for family, family_rows in sorted(grouped.items()):
        n = len(family_rows)

        any_diff = sum(
            r["max_gap"] > 0 for r in family_rows
        )

        ff_bf_diff = sum(
            r["FF"] != r["BF"] for r in family_rows
        )

        bf_ab_diff = sum(
            r["BF"] != r["AB"] for r in family_rows
        )

        mean_gap = statistics.mean(
            r["max_gap"] for r in family_rows
        )

        max_gap = max(
            r["max_gap"] for r in family_rows
        )

        print(f"\nFamily: {family}")
        print(f"Instances: {n}")
        print(
            f"Any difference: {any_diff}/{n} "
            f"({100 * any_diff / n:.1f}%)"
        )
        print(f"FF vs BF differ: {ff_bf_diff}/{n}")
        print(f"BF vs AB differ: {bf_ab_diff}/{n}")
        print(f"Mean max bin gap: {mean_gap:.2f}")
        print(f"Largest bin gap: {max_gap}")


def save_results(rows, split):
    if not rows:
        return

    path = OUTPUT_DIR / f"baseline_comparison_{split}.csv"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


if __name__ == "__main__":

    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}.json"

        with open(path, "r") as f:
            dataset = json.load(f)

        rows = compare_baselines(dataset)

        print_summary(rows, split)
        save_results(rows, split)
