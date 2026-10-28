# testing the evaluator
from src.evaluation import evaluate
from src.baselines import (
    first_fit,
    best_fit,
    make_ab_first_fit
)

heuristics = {
    "First Fit": first_fit,
    "Best Fit": best_fit,
    "ab-FirstFit": make_ab_first_fit(11, 35)
}

for name, heuristic in heuristics.items():

    print(f"\n===== {name} =====")

    for split in ["train", "val", "test"]:

        result = evaluate(heuristic, split)

        print(f"\n{split.upper()}")
        score = result["score"]
        if score is None:
            print("Score: invalid")
        else:
            print(f"Score: {score:.4f}")
        print(f"Feasible: {result['feasible']}")
        print(f"Mean bins: {result['mean_bins']}")
        print(f"Runtime: {result['total_runtime']:.6f}s")

        for family, score in result["family_scores"].items():
            if score is None:
                print(f"  {family}: invalid")
            else:
                print(f"  {family}: {score:.4f}") # pct excess