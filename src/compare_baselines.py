import json
from pathlib import Path
import pandas as pd
from baselines import simulate, first_fit, best_fit, make_ab_first_fit

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
SPLITS = ('train', 'val', 'test')
AB_A, AB_B = 20, 23


def compare_baselines(dataset):
    """Return per-instance bin counts and pairwise differences."""
    heuristics = {
        'FF': first_fit,
        'BF': best_fit,
        'AB': make_ab_first_fit(AB_A, AB_B),
    }
    rows = []
    for instance_id, instance in dataset.items():
        items = instance['items']
        capacity = instance['capacity']
        if len(items) != instance['num_items']:
            raise ValueError(f'Incorrect num_items for {instance_id}')
        bins = {
            name: simulate(items, capacity, heuristic)
            for name, heuristic in heuristics.items()
        }
        rows.append({
            'instance_id': instance_id,
            'family': instance['family'],
            'num_items': len(items),
            **bins,
            'ff_bf_diff': bins['FF'] - bins['BF'],
            'bf_ab_diff': bins['BF'] - bins['AB'],
            'ff_ab_diff': bins['FF'] - bins['AB'],
            'max_gap': max(bins.values()) - min(bins.values()),
        })
    return pd.DataFrame(rows)


def analyse_baselines(df):
    """Summarize discrimination and relative improvements by family."""
    if df.empty:
        raise ValueError('No instances to analyse')
    df = df.copy()
    df['bf_improvement_pct'] = 100 * (df['FF'] - df['BF']) / df['FF']
    df['ab_improvement_pct'] = 100 * (df['BF'] - df['AB']) / df['BF']
    return df.groupby('family').agg(
        num_instances=('instance_id', 'count'),
        pct_any_diff=('max_gap', lambda x: 100 * (x > 0).mean()),
        pct_ff_bf_diff=('ff_bf_diff', lambda x: 100 * (x != 0).mean()),
        pct_bf_ab_diff=('bf_ab_diff', lambda x: 100 * (x != 0).mean()),
        mean_ff_bf_gap=('ff_bf_diff', lambda x: x.abs().mean()),
        mean_bf_ab_gap=('bf_ab_diff', lambda x: x.abs().mean()),
        mean_max_gap=('max_gap', 'mean'),
        largest_gap=('max_gap', 'max'),
        bf_improvement_pct=('bf_improvement_pct', 'mean'),
        ab_improvement_pct=('ab_improvement_pct', 'mean'),
    ).round(3)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with (DATA_DIR / f'{split}.json').open('r', encoding='utf-8') as f:
            dataset = json.load(f)
        comparison = compare_baselines(dataset)
        summary = analyse_baselines(comparison)
        comparison.to_csv(OUTPUT_DIR / f'baseline_comparison_{split}.csv', index=False)
        summary.to_csv(OUTPUT_DIR / f'baseline_summary_{split}.csv')
        print(f'\n===== {split.upper()} =====')
        print(summary.to_string())
        print(f'Saved comparison and summary CSVs to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
