# dsa4212-group11
## Setup, data, experiment, how to reproduce results

## Using the Evaluation Script

The `src/evaluation.py` script provides a common evaluation function for all bin-packing heuristics, including baselines, randomly sampled LLM heuristics, and evolved LLM heuristics.

It evaluates heuristics on the same datasets using the same scoring and feasibility rules.

### 1. Heuristic function format

All heuristics must follow this interface:

```python
def my_heuristic(item_size, bin_loads, capacity):
    # Return an existing bin index or None to open a new bin
    return None
```

**Inputs:**
- `item_size`: Size of the current item.
- `bin_loads`: List of current bin loads.
- `capacity`: Maximum bin capacity (100).

**Output:** Return a valid bin index (`int`) or `None` to open a new bin.

The heuristic must not modify `bin_loads` or access future items.

### 2. Evaluate a heuristic

Import `evaluate` from `evaluation.py`:

```python
from evaluation import evaluate
from baselines import first_fit

result = evaluate(first_fit, "train")

print(result["score"])
```

The evaluator accepts any heuristic function following the required interface.

### 3. Select a dataset

Specify the dataset using:

```python
evaluate(my_heuristic, "train")  # Training set
evaluate(my_heuristic, "val")    # Validation set
evaluate(my_heuristic, "test")   # Test set
```

You can also pass a JSON file path or an already-loaded dataset dictionary.

**Dataset usage:**
- **Train:** Score candidates during LLM generation/evolution.
- **Validation:** Select the best candidate.
- **Test:** Final generalisation evaluation only. Do not use for tuning.

### 4. Evaluation output

The function returns a dictionary containing:

| Key | Description |
|---|---|
| `score` | Mean normalized bin usage (lower is better) |
| `feasible` | Whether all instances were packed without violating constraints |
| `mean_bins` | Average number of bins used |
| `total_runtime` | Total heuristic execution time across instances |
| `family_scores` | Mean score for each distribution family |
| `results` | Detailed results for every instance |

**Scoring formula:**

(insert later on yes)

The overall score is the average of the family-level mean scores, giving each distribution family equal weight.

Lower scores indicate better performance. Infeasible heuristics receive a score of infinity (`float("inf")`).

### 5. Example: Evaluate an LLM-generated heuristic

```python
from evaluation import evaluate

def candidate_heuristic(item_size, bin_loads, capacity):
    for i, load in enumerate(bin_loads):
        if load + item_size <= capacity:
            return i
    return None

result = evaluate(candidate_heuristic, "train")

print("Score:", result["score"])
print("Feasible:", result["feasible"])
print("Family scores:", result["family_scores"])
```

### 6. Important notes

- All methods must use the same evaluator for fair comparison.
- Do not change the scoring function during experiments.
- Do not use test results to tune or select heuristics.
- The evaluator assumes `simulate()` enforces bin capacity and validates heuristic decisions.
- LLM-generated code should be executed in an isolated process with a timeout before being used in automated experiments.