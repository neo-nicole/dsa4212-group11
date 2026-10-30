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
heuristics. It validates each candidate, runs it in a separate Python process,
applies a timeout, and returns structured JSON results. The default
`run_candidate` path uses one worker for the complete dataset. An optional
per-instance path starts a fresh worker and timeout for each individual
instance.

### Candidate function format

Candidates evaluated by the sandbox must define exactly one function named
`priority_fn`:

```python
def priority_fn(item, remaining, capacity):
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
    -> src.evaluation.evaluate(...) or evaluate_instance(...)
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
`src.evaluation.py`, which uses the shared simulator in
`src.baselines.py`. Whole-dataset mode calls `evaluate`; per-instance mode
calls `evaluate_instance`, and the parent aggregates the individual results.

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

For FunSearch-style per-instance isolation, add `--per-instance`:

```powershell
python -m src.sandbox --candidate candidate.py --split train --timeout 10 --per-instance
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
def priority_fn(item, remaining, capacity):
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

Use `run_candidate_per_instance` when each individual instance should have
its own worker and timeout:

```python
from src.sandbox import run_candidate_per_instance

result = run_candidate_per_instance(
    candidate_source,
    load_dataset("train"),
    timeout=10,
)
```

In `run_candidate`, the timeout covers the complete dataset evaluation. In
`run_candidate_per_instance`, it applies independently to each complete
bin-packing instance. Both functions return the same aggregate result shape.

Invalid candidates should be rejected. For example, a candidate that returns a scalar instead of the required score array should produce `status: "invalid"`. A candidate that never returns should produce `status: "timeout"` when run with a sufficiently short timeout.

## Using the OpenRouter API Key and LLM Client

The `src/llm_client.py` module sends prompts to the LLM through OpenRouter. It logs every call (tokens, cost, latency, prompt and response) and tracks spending against the key's budget.

Method A, Method B and the pilot must all call the LLM through this module, so that every run uses the same model settings and is logged the same way.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The client needs `requests` and `python-dotenv` (both listed in `requirements.txt`).

### 2. Set up your API key

Create a file named `.env` in the project root (the same folder as `README.md`) containing:

```text
OPENROUTER_API_KEY=sk-or-v1-...
```

- `.env` is listed in `.gitignore` and should not be commited. 

### 3. Test the setup

From the project root, run:

```bash
python -m src.llm_client
```

This should prints:

- the credit left on your key, before and after the call;
- the reply (should be `hello`);
- the token usage and cost of the call;
- the totals for the `smoke_test` run, and the budget status.

### 4. Fixed settings

These settings are defined at the top of `src/llm_client.py` and must stay consistent for Method A and Method B:

| Setting | Value | Reason |
|---|---|---|
| `MODEL` | `google/gemini-3.6-flash-20260721` | Dated version, so the model cannot change mid-project |
| `TEMPERATURE` | `1.0` | Method A relies on sampling for diversity |
| `MAX_TOKENS` | `2000` | Caps output length; to be adjusted after the pilot |
| Reasoning effort | `low` | Thinking tokens are billed as output and count toward the token budget |
| `BUDGET_USD` | `15.0` | The key's spending limit, used by the budget safety net |
| `WARN_AT` | `0.8` | Print a warning once 80% of `BUDGET_USD` is spent |

### 5. Make a single call

```python
from src.llm_client import call_llm

out = call_llm(
    "Write hello world in Python",
    run_id="debug_hello",
    method="test",
)

print(out["text"])          # the LLM's reply
print(out["total_tokens"])  # prompt + completion tokens
print(out["cost"])          # USD
```

`call_llm` retries failed requests up to 3 times. If every retry fails, `out["text"]` is `None` and `out["error"]` contains the last error.

### 6. Use the client in an experiment loop

Each run stops when it reaches its token budget `T`, which is the same for Method A and Method B. `check_budget()` is only a safety net on the key's total dollar spend. It prints a warning but does not stop the run, so the loop must check it:

```python
from src.llm_client import call_llm, check_budget, run_totals

tokens_used = 0
while tokens_used < TOKEN_BUDGET:              # experiment budget T
    if check_budget()["left"] <= 0.5:          # stop slightly early: spend can lag
        print("Key budget almost used up - stopping run")
        break
    out = call_llm(prompt, run_id=run_id, method="A", generation=gen)
    tokens_used += out["total_tokens"] or 0
    # extract code -> sandbox -> fitness

print(run_totals(run_id))                      # calls, tokens, cost, latency for this run
```

### 7. What is logged

Every call appends one JSON line to `outputs/llm_logs/<run_id>.jsonl`, including failed calls:

| Field | Description |
|---|---|
| `time` | When the call finished |
| `run_id` | Run the call belongs to (also the log file name) |
| `method` | `"A"`, `"B"` or `"test"` |
| `generation` | Method B round number (`0` = seed population); `None` for Method A and tests |
| `model`, `temperature` | Settings used for the call |
| `prompt_tokens` | Input tokens sent |
| `completion_tokens` | Output tokens (includes `reasoning_tokens`) |
| `reasoning_tokens` | Hidden thinking tokens, a subset of `completion_tokens` (`None` if not reported) |
| `total_tokens` | `prompt_tokens + completion_tokens`; counts toward the token budget |
| `cost` | USD for the call (`None` if unknown) |
| `latency_s` | Seconds from first attempt to finish, including retries |
| `attempts` | Number of tries (`>1` means the call was retried) |
| `error` | Last error message if every retry failed, else `None` |
| `prompt`, `response` | Full prompt sent and full reply received |

`run_totals(run_id)` sums the calls, tokens, cost and latency in a run's log.

**Run naming:**

- Pilot and reported runs: e.g. `pilot_A`, `pilot_B`, `runA_1`, `runB_2`. 

### 8. Budget and credit

| Function | Returns |
|---|---|
| `credit_remaining()` | Dollars left on the key |
| `check_budget()` | Total spent on the key vs `BUDGET_USD`, as a dict with `spent`, `budget`, `left`, `fraction_used` |
| `run_totals(run_id)` | Calls, tokens, cost and latency for one run, plus the share of the budget it used |

### 9. Important notes

- All LLM calls for experiments must go through `call_llm`, so that every call is logged.
- Use `total_tokens` (not only `completion_tokens`) for the token budget: Method B's extra cost comes mostly from longer prompts.
- Do not commit a log file while its run is still writing to it.
- The API key is sent only in request headers and is never written to the logs.
