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
evaluate(my_heuristic, "in-test")  # In-distribution test set
evaluate(my_heuristic, "out-test") # Out-of-distribution test set
```

You can also pass a JSON file path or an already-loaded dataset dictionary.

**Dataset usage:**
- **Train:** Used for training-related experiments and development.
- **Validation:** Used for pilot experiments, candidate comparison, and selection.
- **In-test:** Final evaluation on the same distributions as training.
- **Out-test:** Final generalisation evaluation on unseen distributions only. Do not use for tuning.

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

The project supports two heuristic interfaces. The standard evaluation interface is used by baseline heuristics and selects a bin directly. LLM-generated candidates use a separate `priority_fn` interface, which assigns priority scores to feasible bins and the option of opening a new bin. The sandbox converts these scores into bin-placement decisions during evaluation.

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
    -> src.simulator.simulate(...)
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
`src.simulator.py`. Whole-dataset mode calls `evaluate`; per-instance mode
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

Can select `train`, `val`, `in-test`, or `out-test` with `--split`.

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

## Method A and Method B

### Shared setup

- **Starting program:** Both methods start from Best Fit. It is evaluated before any LLM call and logged as generation 0 (0 tokens, no parents). During the pilot, Best Fit and all generated candidates are evaluated on the training set (train). The validation set (val) is reserved for evaluating and comparing candidate heuristics and experimental configurations after evolution.
- **Prompt:** Both methods use the same prompt format. Each prompt includes the problem specification (rules, function interface, constraints), existing program(s) for LLM to improve on, and the start of the next version to be completed by the LLM
- **Extraction:**  `priority_fn_v{k}` is renamed back to `priority_fn` for sandbox to run it. Replies without that function, or with invalid syntax, are logged as invalid.
- **Score:** A candidate's score is its validation score from the sandbox: % excess over the lower bound, averaged within each family and then across families (where lower is better).

Note: the LLM is never shown any scores. With two programs, it can only tell which is better from the version numbers (`priority_fn_v0` is worse, `priority_fn_v1` is better). Method A always shows only Best Fit (`v0`), Method B shows the 2 selected parents, except on its first call, when only Best Fit exists.

### Method A: independent sampling

Every call uses the same prompt, `INITIAL_PROMPT`: Best Fit is shown as `priority_fn_v0`, and LLM is asked to generate `priority_fn_v1`. Later calls never see earlier outputs.

### Method B: LLM-guided evolution

Each call:

1. **Selects parents** from all valid candidates so far, including Best Fit (`select_parents`):
   - candidates with identical per-instance performance signatures count once, keeping the shortest code
   - each signature is the ordered tuple of finite per-instance percentage-excess scores
   - 2 parents are drawn with probability proportional to `exp(-score/T)` (better candidates have higher chance of getting selected), with replacement
   - they are sorted from worse to better, so the better one becomes `priority_fn_v1`.
   On the first call, the prompt has Best Fit as the only parent (identical to Method A).
2. **Builds the prompt** with `generate_prompt(parents)` and asks for the next version.
3. **Evaluates the new program** on the training set (% excess) and logs its signature, along with which parents were in its prompt.
4. **Adds the child to the valid candidates** if it runs successfully, so it can be chosen as a parent later. Invalid candidates are logged but will not be selected to become parents.

### Pilot Run Settings (`src/run_pilot.py`)
- `NUM_CALLS`: 300 LLM calls per method in the pilot
- `FUNCTIONS_PER_PROMPT`: 2 parents per prompt
- `SELECTION_TEMPERATURE`: 0.1 (follows FunSearch's default value)
- `RANDOM_SEED`: 0 for parent selection
- `DATASET`: `train` (training set for evolutionary feedback)
- `BUDGET_USD`: 15.0 (shared experiment budget across both methods)

The Method B selection temperature can be overridden from the command line:

```bash
python3 -m src.run_pilot --method B --selection-temperature 0.1
python3 -m src.run_pilot --method B --selection-temperature 0.5
python3 -m src.run_pilot --method B --selection-temperature 1.0
```

Each summary records the selected temperature and other experiment settings.
The standalone `src.run_method_b` launcher uses the module default and does not
accept this command-line override.

### Output files

- `outputs/pilot/candidates_<run_id>.jsonl`: one line per candidate, starting with Best Fit (generation 0). Recorded fields include: API fields (tokens, cost, latency), evaluation result, ordered per-instance `signature`, `parent_ids`, `parent_scores` (% excess), `num_parents` and `cumulative_tokens`.
- `outputs/pilot/best_candidates/<run_id>.py`: best candidate of the run
- `outputs/pilot/summary_<timestamp>.json`: per-method summary, including the starting program's score.
- `outputs/pilot/pilot_curves.png`: Actual candidate scores against cumulative token usage, with rolling averages.
- `outputs/pilot/best_so_far_curves.png`: Best-so-far performance against cumulative token usage. 

### Generate pilot plots

After running the pilot, generate the performance plots from the saved candidate files:

```bash
python3 -m src.plot_pilot
```

## Using the OpenRouter API Key and LLM Client

The `src/llm_client.py` module sends prompts to the LLM through OpenRouter. It logs every call (tokens, cost, latency, prompt and response) and tracks spending against the key's budget.

Method A, Method B and the pilot must all call the LLM through this module, so that every run uses the same model settings and is logged the same way.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The client needs `requests` and `python-dotenv`.

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

This checks the OpenRouter configuration and displays the configured model and available budget information.

A separate smoke test can be performed using `call_llm()` to verify that an actual API request succeeds. Smoke-test calls are logged and incur API costs.

### 4. Fixed settings

These settings are defined at the top of `src/llm_client.py` and must stay consistent for Method A and Method B:

| Setting | Value | Reason |
|---|---|---|
| `MODEL` | `google/gemini-3.8-flash` | LLM used for candidate generation |
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

The experiment runners use `call_llm()` to send requests and record token usage, cost and latency. `BUDGET_USD` provides a spending safeguard, while `NUM_CALLS` in `src/run_pilot.py` controls the number of LLM calls per method (300 for the pilot). How Method A and Method B build their prompts is described in [Method A and Method B](#method-a-and-method-b).

To run the experiment from the project root:

```bash
python3 -m src.run_pilot              # both methods
python3 -m src.run_pilot --method A   # or: python3 -m src.run_method_a
python3 -m src.run_pilot --method B   # or: python3 -m src.run_method_b
python3 -m src.run_pilot --local-test # sandbox check, no API calls
```

Running these commands (except `--local-test`) makes actual API calls and incurs costs.

`src/resume_method_b.py` was written for the pilot_old, which used the old Method B design. It is not compatible with the current code and should not be used.

### 7. What is logged

Every call appends one JSON line to `outputs/llm_logs/<run_id>.jsonl`, including failed calls:

| Field | Description |
|---|---|
| `time` | When the call finished |
| `run_id` | Run the call belongs to (also the log file name) |
| `method` | `"A"`, `"B"` or `"test"` |
| `generation` | Call number within the run (1, 2, ...) for both methods; `None` for tests |
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

`run_totals(run_id)` summarises calls, token usage, cost, latency, retried calls and the fraction of the configured budget used for a run.

**Run naming:**

- Pilot and reported runs: e.g. `pilot_A`, `pilot_B`, `runA_1`, `runB_2`. 

### 8. Budget and credit

| Function | Returns |
|---|---|
| `credit_remaining()` | Remaining limit reported by the OpenRouter API key |
| `check_budget()` | Budget usage information relative to the configured `BUDGET_USD` limit |
| `run_totals(run_id)` | Number of calls, prompt/completion/total tokens, cost, latency, retried calls and fraction of the configured budget used for one run |

`BUDGET_USD` is the experiment's configured spending limit. It should not be confused with the actual prepaid credit or spending limits on the OpenRouter account.

`check_budget()` compares the API key's reported usage against `BUDGET_USD`. It prints a warning when usage reaches `WARN_AT` (80%) or exceeds the configured budget. It does not automatically stop experiment loops; stopping conditions must be implemented separately in the experiment runner.

### 9. Important notes

- All LLM calls for experiments must go through `call_llm`, so that every call is logged.
- Use `total_tokens` (not only `completion_tokens`) for the token budget: Method B's extra cost comes mostly from longer prompts.
- Do not commit a log file while its run is still writing to it.
- The API key is sent only in request headers and is never written to the logs.
