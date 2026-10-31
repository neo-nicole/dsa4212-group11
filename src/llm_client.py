"""
OpenRouter client with per-call logging.

Setup:
    pip install requests python-dotenv
    put OPENROUTER_API_KEY=sk-or-... in a .env file at the project root

Usage (single call):
    from src.llm_client import call_llm, credit_remaining
    out = call_llm("Write hello world in Python", run_id="smoke", method="test")
    print(out["text"], out["usage"])

Usage (experiment loop):
    from src.llm_client import call_llm, check_budget, run_totals

    tokens_used = 0
    while tokens_used < TOKEN_BUDGET:               # experiment budget (keep the same for A and B)
        if check_budget()["left"] <= 0:             # stop run if exceed safety net
            print("Key budget used up - stopping run")
            break
        out = call_llm(prompt, run_id=run_id, method="A", generation=gen)
        tokens_used += out["total_tokens"] or 0     # failed calls may have billed tokens
        ...                                         # extract code, sandbox, etc

    print(run_totals(run_id))                       # calls, tokens, cost, latency for this run

Notes:
- check_budget() only warns (past WARN_AT), caller has to stop run
- completion_tokens includes thinking (reasoning) tokens
- total_tokens = prompt + completion
- Ensure all settings (MODEL, TEMPERATURE, MAX_TOKENS, reasoning effort) are consistent for Method A and Method B   
"""
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("OPENROUTER_API_KEY")

URL = "https://openrouter.ai/api/v1/chat/completions"
LOG_DIR = ROOT / "outputs" / "llm_logs"

# ---- fixed across Method A and Method B ----
MODEL = "google/gemini-3.8-flash"
PRICE_MODEL = "google/gemini-3.8-flash"
TEMPERATURE = 1.0
MAX_TOKENS = 2000 # adjust after the pilot

# ---- budget (in USD) ----
BUDGET_USD = 15.0  # key's limit
WARN_AT = 0.8      # warn once spending passes this fraction


def _headers():
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not found - check your .env file")
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def credit_remaining():
    """Dollars left on the key (from the course's OpenRouter guide)."""
    r = requests.get("https://openrouter.ai/api/v1/key", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()["data"]["limit_remaining"]


def check_budget():
    """Total spent on the key (across all runs) vs BUDGET_USD; warns past WARN_AT."""
    r = requests.get("https://openrouter.ai/api/v1/key", headers=_headers(), timeout=30)
    r.raise_for_status()
    spent = r.json()["data"]["usage"]
    status = {"spent": spent, "budget": BUDGET_USD, "left": BUDGET_USD - spent,
              "fraction_used": spent / BUDGET_USD}
    if spent >= BUDGET_USD:
        print(f"WARNING: budget used up - spent ${spent:.4f} of ${BUDGET_USD:.2f}")
    elif spent >= WARN_AT * BUDGET_USD:
        print(f"WARNING: spent ${spent:.4f} of ${BUDGET_USD:.2f} ({spent / BUDGET_USD:.0%})")
    return status


_PRICES = {}


def _price(model):
    """(prompt $/token, completion $/token) from OpenRouter's model list, cached per model."""
    if model not in _PRICES:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
        r.raise_for_status()
        for m in r.json()["data"]:
            _PRICES[m["id"]] = (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
    return _PRICES[model]


def _cost(usage, model):
    """Dollar cost of one call: OpenRouter's reported cost, else tokens x listed price."""

    if usage.get("cost") is not None:
        return usage["cost"]

    if usage.get("prompt_tokens") is None:
        return None  # call failed, nothing billed

    try:
        p_in, p_out = _price(PRICE_MODEL)
    except (requests.RequestException, KeyError, ValueError):
        return None  # pricing lookup failed - leave cost unknown

    return (
        usage["prompt_tokens"] * p_in
        + usage.get("completion_tokens", 0) * p_out
    )


def run_totals(run_id):
    """Sum tokens, cost and time over every call logged for run_id."""
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
              "cost": 0.0, "latency_s": 0.0, "retried_calls": 0}
    with open(LOG_DIR / f"{run_id}.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            totals["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost", "latency_s"):
                totals[k] += rec.get(k) or 0
            if (rec.get("attempts") or 1) > 1:
                totals["retried_calls"] += 1
    totals["latency_s"] = round(totals["latency_s"], 2)
    totals["budget_fraction"] = totals["cost"] / BUDGET_USD # share of the budget this run used
    try:
        check_budget()
    except requests.RequestException: # offline - still return the run's totals
        pass
    return totals


def call_llm(prompt, run_id, method, generation=None, model=MODEL, retries=3):
    """
    Send one prompt, return the reply text plus token usage, and append a log line to
    outputs/llm_logs/<run_id>.jsonl. Returns {"text": None, ...} if every retry fails.

    logs:
    - time: time when call is finished
    - run_id (corresponds to log file)
    - method: A (independent sampling) or B (evolution) or test
    - generation: round no., for method B only 
    - model, temperature: settings used
    - prompt_tokens: input tokens sent (prompt, parent code, scores)
    - completion_tokens: output tokens (includes reasoning_tokens)
    - reasoning_tokens: subset of completion_tokens (None if not reported)
    - total_tokens: prompt_tokens + completion_tokens -> counts toward the token budget
    - cost: USD for this call (OpenRouter's reported cost, else tokens x listed price; None if unknown)
    - latency_s: time taken (seconds) for call to finish (includes failed attempts and waits between attempts)
    - attempts: how many tries it took
    - error: last error message if every retry failed, else None
    - prompt: the full prompt sent
    - response: the LLM's full reply text (None if the call failed)

    Returns: dict with text, usage, total_tokens, cost, latency_s, attempts, error. 
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        # Keep thinking low and identical for both methods
        "reasoning": {"effort": "low"},
        "usage": {"include": True}, # return cost of the call
    }

    text, usage, error = None, {}, None
    t0 = time.time()
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers=_headers(), json=body, timeout=120)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})     # prompt_tokens, completion_tokens, ...
            error = None
            break
        except Exception as e: # network error, rate limit, bad response
            error = f"{type(e).__name__}: {e}"
            if attempt < retries - 1:
                time.sleep(2 ** attempt)      # wait 1s, 2s before retrying
    attempts = attempt + 1
    latency_s = round(time.time() - t0, 2)    # includes failed attempts and retry waits

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens") 
    total_tokens = usage.get("total_tokens") # prompt_tokens + completion_tokens
    if total_tokens is None and prompt_tokens is not None:
        total_tokens = prompt_tokens + (completion_tokens or 0)
    cost = _cost(usage, model)

    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "method": method,
        "generation": generation,
        "model": model,
        "temperature": TEMPERATURE,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens, # includes thinking tokens
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "total_tokens": total_tokens, # prompt + completion
        "cost": cost, # USD
        "latency_s": latency_s,
        "attempts": attempts, # if >1 --> latency_s includes retries
        "error": error,
        "prompt": prompt,
        "response": text,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / f"{run_id}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    return {"text": text, "usage": usage, "total_tokens": total_tokens, "cost": cost,
            "latency_s": latency_s, "attempts": attempts, "error": error}


if __name__ == "__main__":
    print("Checking OpenRouter budget information...")

    remaining = credit_remaining()
    input_price, output_price = _price(PRICE_MODEL)

    print(f"Remaining account credit: ${remaining}")
    print(f"Model: {MODEL}")
    print(f"Input price per token: ${input_price}")
    print(f"Output price per token: ${output_price}")
