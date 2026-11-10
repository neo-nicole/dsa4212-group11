# dsa4212-group11

LLM-generated heuristics for online 1D bin packing. Under the same token budget on Gemini 3.8 Flash, we compare:

- **Method A (independent sampling):** every LLM call tries to improve Best Fit from scratch.
- **Method B (LLM-guided evolution):** each call improves on two parents selected from earlier candidates.

The heuristic selected from each run is compared against First Fit, Best Fit and a tuned ab-First-Fit on in-distribution and out-of-distribution test sets.

Group members are listed in [authors.md](authors.md).

## Setup

```bash
pip install -r requirements.txt
```

Only the LLM runs need an API key. Create a `.env` file in the project root (which is git-ignored):

```text
OPENROUTER_API_KEY=sk-or-v1-...
```

Check the key with `python3 -m src.llm_client`.

Run every script as a module **from the project root**, e.g. `python3 -m src.final_eval`.

## Repository structure

```text
data/final_data/        train / val / in-test / out-test splits
src/
  generate_instances.py   generate and load datasets
  simulator.py            online packing simulator
  baselines.py            First Fit, Best Fit, ab-First-Fit
  ab_tuning.py            tune ab-First-Fit's parameters (a, b) on train
  evaluation.py           shared evaluator
  sandbox.py              isolated evaluator for LLM-generated code
  llm_client.py           OpenRouter client with call logging
  prompts/                prompt templates and starting program
  run_pilot.py            Method A / B search loops (pilot)
  run_experiment.py       main experiment
  final_eval.py           selection, testing and analysis
outputs/
  final_experiment/       main experiment candidate logs
  final_eval/final/       final tables, figures and selected heuristics
  pilot_1/, pilot_2/      pilot results
  llm_logs/               raw LLM call logs
report/                   project report
```

Folders named `original_data` or `final_data_before_in_test` are from earlier iterations and are outdated.

## Data

All instances have bin capacity 100 and integer item sizes, generated with fixed seeds (`python3 -m src.generate_instances`).

| Split | Families | Use |
|---|---|---|
| `train` | `uniform_200`, `uniform_500` | search feedback, ABFF tuning |
| `val` | same as train | selecting each run's final heuristic |
| `in-test` | same as train | final evaluation |
| `out-test` | Weibull, large items, near-identical, other sizes, sorted orders | generalisation (never used for tuning) |

## Heuristic interface

Every heuristic, baseline or LLM-generated, is a priority function. Best Fit, for example:

```python
def priority_fn(item_size, remaining, capacity):
    gap = remaining - item_size
    return np.append(-gap, -capacity - 1)
```

- `remaining`: remaining capacities of the bins that can fit the item.
- Return one score per bin in `remaining`, plus a final score for opening a new bin. The item goes to the highest score.

**Score:** mean % excess over the lower bound `ceil(sum(items) / capacity)`, averaged per family and then across families. Lower is better.

```python
from src.evaluation import evaluate
from src.baselines import best_fit

print(evaluate(best_fit, "train")["score"])
```

LLM-generated code is run through the sandbox (separate process, timeout, no imports):

```bash
python3 -m src.sandbox --candidate candidate.py --split train --timeout 10
```

## Reproducing the results

```bash
python3 -m src.ab_tuning              # tune ABFF -> outputs/final_data/ab_params.json
python3 -m src.compare_baselines      # baseline results

# main experiment (needs API key): 2 runs, 500k tokens per method, T = 1
python3 -m src.run_experiment --runs 1 --seed 1 --max-tokens 500000 --selection-temperature 1
python3 -m src.run_experiment --runs 1 --seed 2 --max-tokens 500000 --selection-temperature 1

# selection and test evaluation -> outputs/final_eval/final/
python3 -m src.final_eval --runs-dir outputs/final_experiment --prefix final
```

LLM outputs are not deterministic. To reproduce the reported tables and figures exactly, run only the last command on the committed logs in `outputs/final_experiment/`.

## Results

`outputs/final_eval/final/` contains:

- `table_excess_by_family.csv`: % excess per test family for each heuristic
- `table_bins_vs_bestfit.csv`: bins used relative to Best Fit (negative = fewer bins)
- `table_cost.csv`: tokens, cost and runtime per run
- `fig_*.png`: plots of the search and test results
- `heuristics/`: code of the heuristic selected from each run
