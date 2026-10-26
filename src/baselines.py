
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

    print("All tests passed!")


# ============================================================
# Example Run
# ============================================================
if __name__ == "__main__":
    test_baselines()

    items = [27, 82, 18, 42, 25]
    capacity = 100

    ff_result = simulate(items, capacity, first_fit)
    bf_result = simulate(items, capacity, best_fit)

    print("Items:", items)
    print("Bin capacity:", capacity)
    print("First Fit bins used:", ff_result)
    print("Best Fit bins used:", bf_result)
