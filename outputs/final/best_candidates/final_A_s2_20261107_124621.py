def priority_fn(item_size, open_bin_remaining_capacities, capacity):
    """Improved version of `priority_fn_v0`."""
    rem = np.asarray(open_bin_remaining_capacities, dtype=float)
    if rem.size == 0:
        return np.array([0.0])

    gaps = rem - item_size
    # Exact fits get the highest priority
    exact_bonus = np.where(gaps == 0, 1e6, 0.0)
    # Prefer tighter fits (Best Fit baseline), but penalise leaving awkward tiny residual gaps
    # that cannot fit common future items unless it's an exact fit (gap == 0).
    fit_score = -gaps
    gap_penalty = np.where((gaps > 0) & (gaps < 0.1 * capacity), -5.0, 0.0)

    # Prefer bins that are already well-filled to close bins faster
    filled_ratio = (capacity - rem) / capacity

    scores = exact_bonus + fit_score + 2.0 * filled_ratio + gap_penalty
    new_bin_score = -capacity - 1.0

    return np.append(scores, new_bin_score)