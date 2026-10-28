"""
Instance generator for 1D bin packing.

Usage:
    python generate_instances.py # creates /data/{train,val,test}.json

In other scripts:
    from generate_instances import load_dataset, by_family
    datasets = load_dataset("train")  # {instance_name: instance}
    for family, instances in by_family(datasets).items():
        for inst in instances:
            items, cap, lb = inst["items"], inst["capacity"], inst["lower_bound"]
"""

import json
import math
import os
from collections import defaultdict

import numpy as np

CAPACITY = 100 # fix bin capacity to 100
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")  # default output folder
SPLIT_FILES = {"train": "train.json", "val": "val.json", "test": "test.json"}


def uniform(rng, size, low=1, high=100):
    return rng.integers(low, high + 1, size=size).tolist()


def weibull(rng, size, shape=3.0, scale=45.0):
    """
    Generate a list of integers from a Weibull distribution, clipped to [1, 100].
    Note: parameters shape = 3.0, scale = 45.0 are chosen to
    roughly match the FunSearch Weibull distribution.
    """
    raw = rng.weibull(shape, size=size) * scale
    return np.clip(np.rint(raw), 1, CAPACITY).astype(int).tolist()


def sorted_desc(rng, size):
    return sorted(uniform(rng, size), reverse=True)


def sorted_asc(rng, size):
    return sorted(uniform(rng, size))


def make_instance(family, items):
    return {
        "family": family,
        "capacity": CAPACITY,
        "num_items": len(items),
        "items": items,
        "lower_bound": math.ceil(sum(items) / CAPACITY), # can remove if prefer to compute in evaluate script
    }


def add_family(store, prefix, family, n_instances, base_seed, sampler):
    """Add n_instances instances; instance i uses seed base_seed + i."""
    for i in range(n_instances):
        rng = np.random.default_rng(base_seed + i)
        store[f"{prefix}_{i:02d}"] = make_instance(family, sampler(rng))


def generate_datasets():
    datasets = {"train": {}, "val": {}, "test": {}}

    # Training set : 20 instances each
    tr = datasets["train"]
    add_family(tr, "train_unif_200",  "uniform_200",    20, 33,  lambda r: uniform(r, 200))
    add_family(tr, "train_small_200", "small_1_30_200", 20, 133, lambda r: uniform(r, 200, 1, 30))
    add_family(tr, "train_unif_500",  "uniform_500",    20, 233, lambda r: uniform(r, 500))

    # Validation set: same distributions as training, but 10 instances each
    va = datasets["val"]
    add_family(va, "val_unif_200",  "uniform_200",    10, 1033, lambda r: uniform(r, 200))
    add_family(va, "val_small_200", "small_1_30_200", 10, 1133, lambda r: uniform(r, 200, 1, 30))
    add_family(va, "val_unif_500",  "uniform_500",    10, 1233, lambda r: uniform(r, 500))

    # Test set: 20 instances each
    te = datasets["test"]
    add_family(te, "test_weibull_200",   "weibull_200",          20, 2033, lambda r: weibull(r, 200))
    add_family(te, "test_large_200",     "large_50_100_200",     20, 2133, lambda r: uniform(r, 200, 50, 100))
    add_family(te, "test_identical_200", "near_identical_40_45", 20, 2233, lambda r: uniform(r, 200, 40, 45))
    add_family(te, "test_unif_100",      "uniform_100",          20, 2333, lambda r: uniform(r, 100))
    add_family(te, "test_unif_1000",     "uniform_1000",         20, 2433, lambda r: uniform(r, 1000))

    # for sorted, used same seed so that can ensure diff in performance is due to ordering, not underlying items
    add_family(te, "test_sorted_desc_200", "sorted_largest_first_200",  20, 2533, lambda r: sorted_desc(r, 200))
    add_family(te, "test_sorted_asc_200",  "sorted_smallest_first_200", 20, 2533, lambda r: sorted_asc(r, 200))

    return datasets


# helper functions
def load_dataset(name, folder=DATA_DIR):
    """ Load train/val/test datasets.
    name is 'train', 'val', or 'test'.
    """
    with open(os.path.join(folder, f"{name}.json")) as f:
        return json.load(f)


def by_family(split):
    """Group a split ({name: instance}) into {family: [instances]}.
    e.g. by_family(load_dataset("train")) -> {"uniform_200": [...], "small_1_30_200": [...], ...}
    """
    groups = defaultdict(list)
    for inst in split.values():
        groups[inst["family"]].append(inst)
    return dict(groups)


if __name__ == "__main__":
    datasets = generate_datasets()
    os.makedirs(DATA_DIR, exist_ok=True)
    for split, data in datasets.items():
        path = os.path.join(DATA_DIR, SPLIT_FILES[split])
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        # Note: expect 60, 30, 140 instances for train, val, test respectively
        print(f"Created {split} dataset with {len(data)} instances.")
        # print summary of each split for quick check
        print(f"{split}: {len(data)} instances -> {path}")
        for family, insts in by_family(data).items():
            mean_size = np.mean([np.mean(i["items"]) for i in insts])
            print(f"  {family:28s} {len(insts):3d} inst | {insts[0]['num_items']:5d} items | mean size {mean_size:5.1f}")
        print()