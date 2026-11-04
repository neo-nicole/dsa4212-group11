def priority_fn(item_size, open_bin_remaining_capacities, capacity):
    """Improved version of `priority_fn_v0`."""
    rem = np.asarray(open_bin_remaining_capacities, dtype=float)
    if rem.size == 0:
        return np.array([0.0])

    gaps = rem - item_size
    # Perfect fit gets a large bonus
    perfect_fit_bonus = np.where(gaps == 0, 1000.0, 0.0)

    # Penalise leaving very small awkward gaps that can rarely be filled
    # especially gaps in the range [1, 9] which often waste bin capacity
    small_gap_penalty = np.where((gaps > 0) & (gaps < 0.1 * capacity), -5.0, 0.0)

    # Base score favors smaller remaining gaps (Best Fit principle)
    # plus slight preference for bins that are already fuller
    scores = -gaps + perfect_fit_bonus + small_gap_penalty + 0.01 * (capacity - rem)

    # Score for opening a new bin: always worse than any feasible placement
    new_bin_score = -capacity - 10.0
    return np.append(scores, new_bin_score)