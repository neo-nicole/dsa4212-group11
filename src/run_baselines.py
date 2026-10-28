#testing the evaluator too
from evaluation import evaluate
from baselines import (
    first_fit,
    best_fit,
    make_ab_first_fit
)

heuristics = {
    "First Fit": first_fit,
    "Best Fit": best_fit,
    "ab-FirstFit": make_ab_first_fit(20, 23)
}

for name, heuristic in heuristics.items():

    print(f"\n===== {name} =====")

    for split in ["train", "val", "test"]:

        result = evaluate(heuristic, split)

        print(f"\n{split.upper()}")
        print(f"Score: {result['score']:.4f}")
        print(f"Feasible: {result['feasible']}")
        print(f"Mean bins: {result['mean_bins']}")
        print(f"Runtime: {result['total_runtime']:.6f}s")

        for family, score in result["family_scores"].items():
            print(f"  {family}: {score:.4f}")

