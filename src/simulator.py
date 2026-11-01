import numpy as np


def simulate(items, capacity, priority_fn):
    """Pack items in order and return the number of bins used."""
    remaining = []  # remaining space left in each bin

    for item in items:
        r = np.array(remaining, dtype=float)
        valid = np.nonzero(r >= item)[0]  # bins with enough space for item
        score = np.asarray(
            priority_fn(item, r[valid], capacity),
            dtype=float,
        )
        if score.shape != (len(valid) + 1,) or not np.all(np.isfinite(score)):
            raise ValueError("scores must be len(valid bins)+1 finite numbers")
        k = int(np.argmax(score))
        if k == len(valid):  # highest score is last: open a new bin
            remaining.append(capacity - item)
        else:
            remaining[valid[k]] -= item  # update remaining space

    return len(remaining)
