def priority_fn(item_size, open_bin_remaining_capacities, capacity):
    """Best Fit: the feasible bin with the least space left scores highest."""
    gaps = np.asarray(open_bin_remaining_capacities) - item_size
    return np.append(-gaps, -capacity - 1)