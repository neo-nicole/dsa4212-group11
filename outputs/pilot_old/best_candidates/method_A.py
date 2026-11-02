def priority_fn(item_size, open_bin_remaining_capacities, capacity):
    scores = []
    for rem in open_bin_remaining_capacities:
        new_rem = rem - item_size
        if new_rem == 0:
            score = 10000.0
        else:
            score = 1000.0 - new_rem
        scores.append(score)
    scores.append(0.0)
    return scores