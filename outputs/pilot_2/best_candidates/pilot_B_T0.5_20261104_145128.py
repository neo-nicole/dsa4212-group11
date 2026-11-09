def priority_fn(item_size, open_bin_remaining_capacities, capacity):
    """Calibrated Best-Fit heuristic with exact-closure dominance, item-scaled
    inverse gap power, waste-hazard thresholding, and combinatorial alignment."""
    rem = np.asarray(open_bin_remaining_capacities, dtype=float)
    if rem.size == 0:
        return np.array([0.0])

    cap = float(capacity)
    size = float(item_size)
    gaps = rem - size

    # Scaled Best-Fit power: stronger pressure to pack tightly as item size increases
    p = 2.0 + 0.8 * (size / cap)
    scores = (cap / (gaps + 1.0)) ** p

    # Absolute priority for perfect closure (zero residual gap)
    scores = np.where(gaps == 0.0, scores + cap * 100000.0, scores)

    # Stranded sliver hazard: gaps too small (e.g. 1 to 10% capacity) risk permanent waste,
    # unless gap is actually zero (handled above).
    # Smooth quadratic penalty with steepness near lower bound.
    hazard_limit = 0.10 * cap
    in_hazard = (gaps > 0.0) & (gaps < hazard_limit)
    hazard_penalty = np.where(
        in_hazard,
        ((hazard_limit - gaps) / hazard_limit) ** 1.8 * (cap * 8.0) + (cap * 2.0) / (gaps + 1.0),
        0.0
    )
    scores = scores - hazard_penalty

    # Harmonic resonance / combinatorial flexibility bonus:
    # Prefer leaving gaps that cleanly align with common fractional divisions of capacity
    targets = cap * np.array([0.5, 1.0 / 3.0, 0.25, 0.20, 1.0 / 6.0])
    weights = np.array([4.0, 3.2, 2.4, 1.6, 1.0])
    widths = cap * np.array([0.040, 0.032, 0.025, 0.020, 0.015])
    diffs = gaps[:, np.newaxis] - targets[np.newaxis, :]
    harmonic_bonus = np.sum(weights * np.exp(-0.5 * (diffs / widths) ** 2), axis=1)
    scores = scores + harmonic_bonus

    # Consolidation bias: break ties in favor of bins already filled more deeply
    fullness = (cap - rem) / cap
    scores = scores + (fullness ** 1.5) * 5.0

    # Ensure opening a new bin is strictly suboptimal when a feasible bin exists
    new_bin_score = -1e12

    return np.append(scores, new_bin_score)