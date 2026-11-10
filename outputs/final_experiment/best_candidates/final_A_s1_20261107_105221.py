def priority_fn(item_size, open_bin_remaining_capacities, capacity):
    """Improved heuristic balancing tight fits and avoiding unusable tiny residual gaps."""
    rem = np.asarray(open_bin_remaining_capacities, dtype=float)
    if rem.size == 0:
        return np.array([0.0])

    new_gaps = rem - item_size
    # Exact fit is given the highest priority bonus
    exact_fit_bonus = np.where(new_gaps == 0, 1000.0, 0.0)

    # Base score favors smaller remaining gaps (Best Fit principle)
    # Penalize leaving very small, awkward residuals (< 0.1 * capacity) that cannot fit most items
    awkward_residual_penalty = np.where((new_gaps > 0) & (new_gaps < 0.1 * capacity), 5.0, 0.0)

    # Favor filling bins that are already well-packed to completely retire bins
    fullness_reward = (capacity - rem) / capacity

    scores = -new_gaps + exact_fit_bonus - awkward_residual_penalty + fullness_reward
    new_bin_score = -capacity - 100.0
    return np.append(scores, new_bin_score)