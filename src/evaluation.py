import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from src.baselines import best_fit, first_fit, make_ab_first_fit, simulate

#config
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "final_data"
OUTPUT_DIR = ROOT / 'outputs' / 'final_data'

SPLITS = {
    "train": DATA_DIR / "train.json",
    "val": DATA_DIR / "val.json",
    "test": DATA_DIR / "test.json"
}

#load datasets
def load_instances(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def resolve_dataset(dataset):
    """
    Accept:
    1. Split name: "train", "val", "test"
    2. JSON filepath: "data/train.json"
    3. Already-loaded dataset dictionary
    """

    if isinstance(dataset, dict):
        return dataset

    if isinstance(dataset, (str, Path)):
        if str(dataset).lower() in SPLITS:
            return load_instances(
                SPLITS[str(dataset).lower()]
            )

        return load_instances(dataset)

    raise TypeError(
        "dataset must be a split name, "
        "JSON filepath, or dictionary"
    )


#eval heuristic
#eval one instance
def evaluate_instance(heuristic, instance_id, instance):

    items = instance["items"]
    capacity = instance["capacity"]
    family = instance["family"]

    if len(items) != instance["num_items"]:
        raise ValueError(
            f"Incorrect num_items for {instance_id}"
        )

    if type(capacity) is not int or capacity <= 0:
        raise ValueError("Invalid bin capacity")

    if any(
        type(item) is not int or not 1 <= item <= capacity
        for item in items
    ):
        raise ValueError(
            f"Invalid item size in {instance_id}"
        )

    start = time.perf_counter()

    try:
        bins_used = simulate(items, capacity, heuristic)
        feasible = True
        error = None

    except Exception as e:
        bins_used = None
        feasible = False
        error = str(e)

    runtime = time.perf_counter() - start

    lower_bound = math.ceil(sum(items) / capacity)

    if feasible:
        if lower_bound < 0:
            raise ValueError("lower_bound cannot be negative")

        if lower_bound == 0:
            if bins_used != 0:
                raise ValueError(
                    "An instance with zero lower bound must use zero bins"
                )
            score = 0.0
        else:
            score = 100 * (bins_used - lower_bound) / lower_bound
    else:
        score = None

    return {
        "instance_id": instance_id,
        "family": family,
        "num_items": len(items),
        "status": "ok" if feasible else "invalid",
        "feasible": feasible,
        "bins_used": bins_used,
        "lower_bound": lower_bound,
        "score": score,
        "runtime": runtime,
        "error": error
    }

#eval on whole dataset
def evaluate(heuristic, dataset="train"):
    """
    Evaluate any heuristic on a dataset.

    heuristic:
        Function with signature:
        heuristic(item_size, remaining, capacity)

    dataset:
        "train", "val", "test",
        JSON filepath, or dataset dictionary.

    Returns:
        Dictionary of evaluation metrics.
    """

    dataset = resolve_dataset(dataset)

    if not dataset:
        raise ValueError("Dataset is empty")

    results = []

    for instance_id, instance in dataset.items():
        result = evaluate_instance(
            heuristic,
            instance_id,
            instance
        )
        results.append(result)

    grouped = defaultdict(list)

    for result in results:
        grouped[result["family"]].append(result)

    family_scores = {}

    for family, family_results in grouped.items():

        if any(
            not r["feasible"]
            for r in family_results
        ):
            family_scores[family] = None

        else:
            family_scores[family] = statistics.mean(
                r["score"]
                for r in family_results
            )

    feasible = all(
        r["feasible"] for r in results
    )

    overall_score = (
        statistics.mean(family_scores.values())
        if feasible else None
    )

    return {
        "status": "ok" if feasible else "invalid",
        "feasible": feasible,
        "score": overall_score,
        "mean_bins": (
            statistics.mean(
                r["bins_used"] for r in results
            )
            if feasible else None
        ),
        "total_runtime": sum(
            r["runtime"] for r in results
        ),
        "family_scores": family_scores,
        "results": results
    }