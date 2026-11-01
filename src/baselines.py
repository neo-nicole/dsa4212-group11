import json
import os
from pathlib import Path

import numpy as np

from src.simulator import simulate


# ============================================================
# First Fit
# ============================================================
def first_fit(item_size, remaining, capacity):
    """Earliest bin scores the highest
        remaining: space left in each bin that the item can fit in, arranged in bin order """
    bins = -np.arange(len(remaining), dtype=float)
    return np.append(bins, -len(remaining) - 1) #new bin scores the worst 

# ============================================================
# Best Fit (BF) 
# ============================================================
def best_fit(item_size, remaining, capacity):
    """
    tightest fit scores the highest
    """
    bins = -(remaining - item_size) #negative of remaining space in each bin after placing
    return np.append(bins, -capacity - 1) #new bin scores the worst


# ============================================================
# ab First Fit
# ============================================================
def make_ab_first_fit(a, b):
    """
    gap: space left in the bin after placing item.
    1. if any bin has gap in [0,a] - use best fit
    2. if any bin has gap>=b - use first fit
    3. else - open a new bin
    skip bins with gap in (a,b)
    """
    assert float(a).is_integer() and float(b).is_integer(), "a, b must be integers"
    assert 0 <= a < b, "need 0 <= a < b" 

    def ab_first_fit(item_size, remaining, capacity):
        gap = remaining - item_size
        n = len(remaining)
        tight = 2 * capacity - gap                        
        loose = capacity - np.arange(n) / (n + 1)        
        bins = np.where(gap <= a, tight, np.where(gap >= b, loose, -1.0))  # skipped = -1
        return np.append(bins, 0.0)                       # new bin scored 0.0 (beats skipped bins)

    ab_first_fit.params = (a, b)
    return ab_first_fit

def load_tuned_ab_first_fit(path="outputs/final_data/ab_params.json"):
    """Build the ab-ff from best params (a, b) found through tuning."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path) as f:
        p = json.load(f)
    return make_ab_first_fit(p["a"], p["b"])

# ============================================================
# Test
# ============================================================
def test_baselines():
    capacity = 100

    test_cases = [
        ([], 0),
        ([50], 1),
        ([100], 1),
        ([40, 60], 1),
        ([60, 60, 60], 3),
    ]

    for items, expected in test_cases:
        assert simulate(items, capacity, first_fit) == expected
        assert simulate(items, capacity, best_fit) == expected

    # scoring fns return one score per valid bin + one for a new bin
    for fn in (first_fit, best_fit, make_ab_first_fit(a=2, b=10)):
        assert fn(30, np.array([50.0, 30.0]), capacity).shape == (3,)
        assert fn(30, np.array([]), capacity).shape == (1,)

    # first fit picks the earliest bin, best fit picks the tightest bin
    assert choose(first_fit, 30, [50, 30], capacity) == 0
    assert choose(best_fit, 30, [50, 30], capacity) == 1
    assert choose(first_fit, 30, [], capacity) is None
    assert choose(best_fit, 30, [], capacity) is None

    # same as best fit when a>=capacity
    ab_as_bf = make_ab_first_fit(a=capacity, b=capacity + 1)
    for items in ([27, 82, 18, 42, 25], [60, 30, 50, 20, 45, 55]):
        assert simulate(items, capacity, ab_as_bf) == simulate(items, capacity, best_fit)

    print("All tests passed!")

def choose(priority_fn, item_size, remaining, capacity):
    """Index of the bin picked from remaining (valid bins only), None if a new bin is opened."""
    remaining = np.array(remaining, dtype=float)
    k = int(np.argmax(priority_fn(item_size, remaining, capacity)))
    return None if k == len(remaining) else k

def test_ab_first_fit():
    capacity = 100
    h = make_ab_first_fit(a=2, b=10)

    # remaining: space left in each valid bin (simulate filters out bins the item can't fit)
    # tight fit wins over loose fit, tightest fit (smallest gap) wins over other tight fits
    assert choose(h, 3, [50, 5, 3], capacity) == 2
    assert choose(h, 3, [50, 3], capacity) == 1
    #loose fit uses first fit
    assert choose(h, 5, [50, 70], capacity) == 0 #first loose bin
    # skip bins with gap in (a, b)
    assert choose(h, 5, [10, 60], capacity) == 1
    assert choose(h, 5, [10], capacity) is None #open new bin
    # boundaries: gap == a is tight, gap == b is loose
    assert choose(h, 5, [7], capacity) == 0
    assert choose(h, 5, [15], capacity) == 0
    assert choose(h, 5, [14], capacity) is None
    #ignore infeasible bins - filtered out by simulate, so 20 opens a 2nd bin
    assert simulate([90, 20], capacity, h) == 2

    print("ab-first-fit tests passed!")

# ============================================================
# Example Run
# ============================================================
if __name__ == "__main__":
    test_baselines()
    test_ab_first_fit()
    
    root = Path(__file__).resolve().parent.parent
    with open(root / "outputs" / "final_data" / "ab_params.json") as f:
        params = json.load(f)

    a, b = params["a"], params["b"]
    print(f"Tuned parameters: a={a}, b={b}")

    abff = make_ab_first_fit(a, b)

    items = [27, 82, 18, 42, 25]
    capacity = 100

    ff_result = simulate(items, capacity, first_fit)
    bf_result = simulate(items, capacity, best_fit)
    abff_result = simulate(items,capacity, abff)

    print("Items:", items)
    print("Bin capacity:", capacity)
    print("First Fit bins used:", ff_result)
    print("Best Fit bins used:", bf_result)
    print("ab-First Fit bins used:", abff_result)
