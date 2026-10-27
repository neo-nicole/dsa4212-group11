import json
import time
import math
import statistics
from collections import defaultdict
from baselines import simulate, first_fit, best_fit

def load_instances(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

#eval heuristic
def evaluate_instance(heuristic, instance_id, instance):

    items = instance["items"]
    capacity = instance["capacity"]
    family = instance["family"]

    #check for dataset consistency
    if len(items) != instance["num_items"]:
        raise ValueError(
            f"Incorrect num_items for {instance_id}"
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

    #theoretical lower bound
    lower_bound = math.ceil(sum(items) / capacity)

    #where lower score is better
    if feasible:
        score = (
            bins_used / lower_bound
            if lower_bound > 0 else 1.0
        )
    else:
        score = float("inf")

    return {
        "instance_id": instance_id,
        "family": family,
        "num_items": len(items),
        "feasible": feasible,
        "bins_used": bins_used,
        "lower_bound": lower_bound,
        "score": score,
        "runtime": runtime,
        "error": error
    }

#eval on whole dataset
def evaluate(heuristic, dataset):

    results = []

    for instance_id, instance in dataset.items():
        result = evaluate_instance(
            heuristic,
            instance_id,
            instance
        )
        results.append(result)

    #group results by distribution family
    grouped = defaultdict(list)

    for result in results:
        grouped[result["family"]].append(result)

    family_scores = {}

    for family, family_results in grouped.items():

        if any(not r["feasible"] for r in family_results):
            family_scores[family] = float("inf")
        else:
            family_scores[family] = statistics.mean(
                r["score"] for r in family_results
            )

    feasible = all(r["feasible"] for r in results)

    # Equal weight to each distribution family
    overall_score = (
        statistics.mean(family_scores.values())
        if feasible and family_scores
        else float("inf")
    )

    return {
        "feasible": feasible,
        "score": overall_score,
        "mean_bins": (
            statistics.mean(r["bins_used"] for r in results)
            if feasible and results else None
        ),
        "total_runtime": sum(r["runtime"] for r in results),
        "family_scores": family_scores,
        "results": results
    }

#run baseline eval
if __name__ == "__main__":

    train = load_instances("..\data\train.json")
    validation = load_instances("..\data\val\.json")
    test = load_instances("..\data\test.json")

    heuristics = {
        "First Fit": first_fit,
        "Best Fit": best_fit
    }

    for name, heuristic in heuristics.items():

        print(f"\n========== {name} ==========")

        for split_name, dataset in [
            ("Train", train),
            ("Validation", validation),
            ("Test", test)
        ]:

            result = evaluate(heuristic, dataset)

            print(f"\n{split_name}")
            print(f"Feasible: {result['feasible']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Mean bins: {result['mean_bins']}")
            print(
                f"Total runtime: "
                f"{result['total_runtime']:.6f}s"
            )

            print("Distribution scores:")

            for family, score in result["family_scores"].items():
                print(f"  {family}: {score:.4f}")