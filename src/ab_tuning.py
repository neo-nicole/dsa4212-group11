"""
Tune ab-First-Fit parameters (a, b) on train split

Run from the project root:
    python -m src.ab_tuning

Outputs:
    outputs/ab_params.json
    outputs/ab_grid.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.generate_instances import load_dataset, by_family
from src.baselines import first_fit, best_fit, make_ab_first_fit
from src.simulator import simulate

A_MAX, B_MAX = 30, 45 # selected arbitrarily, widen search if needed
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "final_data"

def excess_bin_pct(inst, heuristic):
    """
    measure % of extra bins used compared with theoretical lower bound
    """
    used = simulate(inst["items"], inst["capacity"], heuristic)
    lb = inst["lower_bound"]
    return 100 * (used - lb) / lb

def family_scores(groups, heuristic):
    """mean excess % for each family"""
    return [np.mean([excess_bin_pct(inst, heuristic) for inst in insts])
            for insts in groups.values()]

def paired_diff(groups, h1, h2, n_boot=2000, seed=0):
    """
    compare 2 heuristics on same instances
    returns mean difference in excess bin pct, 95% ci for mean difference
    """
    d = np.array([excess_bin_pct(i, h1) - excess_bin_pct(i, h2)
                  for insts in groups.values() for i in insts])
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(boots, [2.5, 97.5])

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load the train split, grouped by family
    train = by_family(load_dataset("train"))
    names = list(train) 

    # grid search
    grid = np.full((len(names), A_MAX + 1, B_MAX + 1), np.nan)
    for a in range(A_MAX + 1):
        for b in range(a + 1, B_MAX + 1):
            grid[:, a, b] = family_scores(train, make_ab_first_fit(a, b))
    
    # score - average across families
    mean_grid = grid.mean(axis=0)
    a_best, b_best = (int(x) for x in np.unravel_index(np.nanargmin(mean_grid), mean_grid.shape))

    # compare with first fit and best fit on same train instances
    ff = float(np.mean(family_scores(train, first_fit)))
    bf = float(np.mean(family_scores(train, best_fit)))
    print(f"best: a={a_best}, b={b_best}  train excess = {mean_grid[a_best, b_best]:.3f}%")
    print(f"first fit = {ff:.3f}%   best fit = {bf:.3f}%")
    if a_best == A_MAX or b_best == B_MAX:
        print("WARNING: best pair is on the grid edge")

    # validation check (sanity check)
    val = by_family(load_dataset("val"))
    v_ab = float(np.mean(family_scores(val, make_ab_first_fit(a_best, b_best))))
    v_ff = float(np.mean(family_scores(val, first_fit)))
    v_bf = float(np.mean(family_scores(val, best_fit)))

    ab_val = make_ab_first_fit(a_best, b_best)
    mean_d, ci = paired_diff(val, ab_val, best_fit)
    print(f"val ab - best fit = {mean_d:+.3f}pp, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"val: ab = {v_ab:.3f}%   first fit = {v_ff:.3f}%   best fit = {v_bf:.3f}%")

    # save params
    with (OUT_DIR / "ab_params.json").open("w", encoding="utf-8") as f:
        json.dump({"a": a_best, "b": b_best, "tuned_on": "train"}, f, indent=2)

    #plot heatmaps - 1 per family + mean
    panels = [(n, grid[k]) for k, n in enumerate(names)] + [("mean", mean_grid)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for ax, (title, g) in zip(axes, panels):
        im = ax.imshow(np.ma.masked_invalid(g), origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_title(title)
        ax.set_xlabel("b")
        ax.set_ylabel("a")
        plt.colorbar(im, ax=ax, label="% excess over lower bound")
        ax.plot(b_best, a_best, "r*", ms=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ab_grid.png", dpi=150)


if __name__ == "__main__":
    main()