import json
from pathlib import Path

# ============================================================
# Bin-Packing Simulator
# ============================================================
def simulate(items, capacity, heuristic):
    """Pack items in order and return the number of bins used."""
    bin_loads = []

    for item in items:
        bin_index = heuristic(item, bin_loads, capacity)

        if bin_index is None:
            bin_loads.append(item)
        else:
            bin_loads[bin_index] += item

    return len(bin_loads)


# ============================================================
# First Fit
# ============================================================
def first_fit(item_size, bin_loads, capacity):
    """Choose the first bin that can fit the item."""
    for i in range(len(bin_loads)):
        if bin_loads[i] + item_size <= capacity:
            return i

    return None


# ============================================================
# Best Fit (BF)
# ============================================================
def best_fit(item_size, bin_loads, capacity):
    """Choose the feasible bin with the least space left afterward."""
    best_bin = None
    least_remaining = float("inf")

    for i in range(len(bin_loads)):
        remaining_after = capacity - bin_loads[i] - item_size

        if remaining_after >= 0 and remaining_after < least_remaining:
            least_remaining = remaining_after
            best_bin = i

    return best_bin


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

    def ab_first_fit(item_size, bin_loads, capacity):
        best_tight, best_gap = None, float("inf")  #idx of best bin satisfying tight fit, smallest gap among tight fitting bins
        first_loose = None                         # idx of first bin satisfying loose fit - gap>=b

        for i in range(len(bin_loads)):
            gap = capacity - bin_loads[i] - item_size  #space left after placing item
            if gap < 0:                                #bin cannot fit item
                continue
            if gap <= a:                               #check tight fit - choose smallest gap (best fit)
                if gap < best_gap:
                    best_gap, best_tight = gap, i
            elif gap >= b and first_loose is None:     #check loose fit - choose first bin that satisfies (first fit)
                first_loose = i

        return best_tight if best_tight is not None else first_loose

    ab_first_fit.params = (a, b)
    return ab_first_fit

def load_tuned_ab_first_fit(path="outputs/ab_params.json"):
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

    # same as best fit when a>=capacity
    ab_as_bf = make_ab_first_fit(a=capacity, b=capacity + 1)
    for items in ([27, 82, 18, 42, 25], [60, 30, 50, 20, 45, 55]):
        assert simulate(items, capacity, ab_as_bf) == simulate(items, capacity, best_fit)

    print("All tests passed!")

def test_ab_first_fit():
    capacity = 100
    h = make_ab_first_fit(a=2, b=10)

    # tight fit wins over loose fit, tightest fit (smallest gap) wins over other tight fits
    assert h(3, [50, 95, 97], capacity) == 2 
    assert h(3, [50, 97], capacity) == 1
    #loose fit uses first fit
    assert h(5, [50, 30], capacity) == 0 #first loose bin 
    # skip bins with gap in (a, b) 
    assert h(5, [90, 40], capacity) == 1
    assert h(5, [90], capacity) is None #open new bin
    # boundaries: gap == a is tight, gap == b is loose
    assert h(5, [93], capacity) == 0             
    assert h(5, [85], capacity) == 0            
    assert h(5, [86], capacity) is None
    #ignore infeasible bins
    assert h(20, [90], capacity) is None

    print("ab-first-fit tests passed!")

# ============================================================
# Example Run
# ============================================================
if __name__ == "__main__":
    test_baselines()
    test_ab_first_fit()
    
    root = Path(__file__).resolve().parent.parent
    with open(root / "outputs" / "ab_params.json") as f:
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
