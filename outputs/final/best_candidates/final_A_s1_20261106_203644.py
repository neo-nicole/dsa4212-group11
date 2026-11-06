def priority_fn(item_size, open_bin_remaining_capacities, capacity):
    """Improved heuristic combining Best Fit with avoidance of awkward residual space."""
    rem = np.asarray(open_bin_remaining_capacities, dtype=float)
    if rem.size == 0:
        return np.array([0.0])

    gaps = rem - item_size
    # Exact fits get the highest priority
    exact_fit_bonus = np.where(gaps == 0, 1e6, 0.0)

    # Base score favors smaller leftover gap (Best Fit)
    base_score = -gaps

    # Penalize bins leaving very tiny, unusable gaps that waste capacity
    unusable_waste_penalty = np.where((gaps > 0) & (gaps < 0.1 * capacity), -5.0, 0.0)

    # Prefer packing larger items into already more filled bins to save empty space for small items
    filling_bonus = (capacity - rem) / capacity

    scores = base_score + exact_fit_bonus + unusable_waste_penalty + filling_bonus
    new_bin_score = -float(capacity) - 1.0

    return np.append(scores, new_bin_score)