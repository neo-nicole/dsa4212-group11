def priority_fn(item_size, open_bin_remaining_capacities, capacity):
    """Dual-regime Best Fit with harmonic alignment, headroom reservation, and waste shaping."""
    rem = np.asarray(open_bin_remaining_capacities, dtype=float)
    if rem.size == 0:
        return np.array([0.0])

    gaps = rem - item_size

    # Base Best-Fit driver: strongly favour tighter post-allocation gaps
    # Hyperbolic term creates asymptotic attraction to exact / near-exact fits
    scores = -gaps * 3.0 + (capacity * 5.0) / (gaps + 1.0)

    # Exceptional bonus for exact closure (gap == 0)
    scores = np.where(gaps == 0.0, scores + capacity * 25.0, scores)

    # Harmonic alignment: reward residual gaps that match common item size quanta
    # If the remaining gap is a multiple of item_size or can neatly host another equal item
    fit_multiple = np.isclose(np.fmod(gaps, item_size), 0.0, atol=1e-5)
    scores = np.where((gaps > 0.0) & fit_multiple, scores + capacity * 0.4, scores)

    # Waste / sliver penalty: gaps smaller than 6% capacity that are unlikely to be filled
    sliver_thresh = 0.06 * capacity
    sliver_penalty = np.where(
        (gaps > 0.0) & (gaps < sliver_thresh),
        ((sliver_thresh - gaps) / sliver_thresh) ** 2 * (capacity * 1.5),
        0.0
    )
    scores -= sliver_penalty

    # Strategic headroom preservation:
    # Discourage placing small items (< 30% capacity) into mostly empty bins (rem >= 70% capacity)
    # to preserve large contiguous space for upcoming large items.
    is_small_item = item_size < (0.30 * capacity)
    is_spacious_bin = rem >= (0.70 * capacity)
    reservation_penalty = np.where(
        is_small_item & is_spacious_bin,
        ((rem - item_size) / capacity) * (capacity * 1.2),
        0.0
    )
    scores -= reservation_penalty

    # Tie-breaking: concentrate mass into already fuller bins
    fullness = (capacity - rem) / capacity
    scores += fullness * 3.0

    # Ensure opening a new bin is strictly disfavoured when feasible placements exist
    new_bin_score = -1e8
    return np.append(scores, new_bin_score)