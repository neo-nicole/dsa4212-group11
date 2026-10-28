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

## Using the Candidate Sandbox

The `src/sandbox.py` module safely evaluates LLM-generated bin-packing
heuristics. It runs each candidate in
a separate Python process, validates the candidate, applies a timeout, and
returns structured JSON results.

### Candidate function format

Candidates evaluated by the sandbox must define exactly one function named
`scoring_fn`:

```python
def scoring_fn(item, remaining, capacity):
    gap = remaining - item
    return np.append(-gap, -capacity - 1)
```

The function receives the current item, the remaining capacities of feasible
existing bins, and the bin capacity. It must return one finite numeric score
for every feasible existing bin, followed by one additional score for opening
a new bin. The shared simulator in `src/baselines.py` chooses the action with
the largest score.

Note: Candidates should not import NumPy. The sandbox provides it as `np`.

### How the sandbox works

The evaluation flow is:

```text
candidate source
    -> syntax and signature validation
    -> isolated worker process
    -> src.baselines.simulate(...)
    -> src.evaluation.evaluate(...)
    -> bins used, percentage excess, and aggregation
    -> JSON result
```

The performance score is:

```python
100 * (bins_used - lower_bound) / lower_bound
```

Lower scores are better. A score of `0.0` means the candidate matched the
theoretical lower bound. The result also includes mean bins, family scores,
runtime, and per-instance results.

The sandbox delegates simulation and metric calculation to
`src.evaluation.evaluate`, which in turn uses the shared simulator in
`src.baselines.py`. This keeps the scoring definition identical for baseline
and generated candidates.

The sandbox reports:

- `ok`: the candidate ran successfully;
- `invalid`: the candidate had invalid syntax, a wrong signature, an exception,
  or invalid output;
- `timeout`: the candidate exceeded the configured time limit.

### Run a candidate file

For a candidate file in the repository root, such as `candidate.py`, run:

```powershell
python -m src.sandbox --candidate candidate.py --split train --timeout 10
```

If the candidate file is stored in `src`, use:

```powershell
python -m src.sandbox --candidate src\candidate.py --split train --timeout 10
```

Can select `train`, `val`, or `test` with `--split`.

### Call the sandbox from Python

If needed, an evolution loop can pass generated source code directly without creating a
candidate file:

```python
from src.sandbox import run_candidate
from src.generate_instances import load_dataset

candidate_source = """
def scoring_fn(item, remaining, capacity):
    gap = remaining - item
    return np.append(-gap, -capacity - 1)
"""

result = run_candidate(
    candidate_source,
    load_dataset("train"),
    timeout=10,
)

print(result["status"])
print(result["score"])
```

Invalid candidates should be rejected. For example, a candidate that returns a scalar instead of the required score array should produce `status: "invalid"`. A candidate that never returns should produce `status: "timeout"` when run with a sufficiently short timeout.