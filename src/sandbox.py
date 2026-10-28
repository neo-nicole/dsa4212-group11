"""Process-isolated runner for LLM-generated bin-packing heuristics.

Candidates must define:

    def scoring_fn(item, remaining, capacity):
        ...

The function returns one finite numeric score for each feasible existing bin,
followed by one score for opening a new bin.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation import evaluate

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "final_data"
SPLITS = {"train", "val", "test"}
WORKER_FLAG = "--sandbox-worker"

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "range": range,
    "sum": sum,
}


class CandidateError(ValueError):
    """Raised when a candidate does not satisfy the sandbox contract."""


def _compile_candidate(source: str):
    """Compile source containing exactly one scoring_fn definition."""
    try:
        tree = ast.parse(source, filename="<candidate>", mode="exec")
    except SyntaxError as exc:
        raise CandidateError(
            f"syntax error: {exc.msg} (line {exc.lineno})"
        ) from exc

    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "scoring_fn"
    ]
    allowed_nodes = [
        node for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    if len(functions) != 1 or len(allowed_nodes) != 1:
        raise CandidateError(
            "candidate must contain exactly one scoring_fn function"
        )
    function = functions[0]
    if isinstance(function, ast.AsyncFunctionDef):
        raise CandidateError("scoring_fn must be a regular function")
    if len(function.args.args) != 3:
        raise CandidateError(
            "scoring_fn must have signature (item, remaining, capacity)"
        )
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        raise CandidateError(
            "imports are not allowed; NumPy is available as np"
        )

    namespace = {"np": np, "__builtins__": SAFE_BUILTINS}
    try:
        compiled = compile(tree, "<candidate>", "exec")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            exec(compiled, namespace)
    except Exception as exc:
        raise CandidateError(f"candidate could not be compiled: {exc}") from exc
    return namespace["scoring_fn"]


def _evaluate(source: str, dataset: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compile a candidate and delegate metrics to the shared evaluator."""
    scoring_fn = _compile_candidate(source)
    return evaluate(scoring_fn, dataset)


def _worker() -> None:
    request = json.load(sys.stdin)
    try:
        response = _evaluate(request["source"], request["dataset"])
    except (CandidateError, ValueError, KeyError) as exc:
        response = {"status": "invalid", "error": str(exc)}
    print(json.dumps(response, allow_nan=False))


def run_candidate(
    source: str,
    dataset: dict[str, dict[str, Any]],
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Evaluate candidate source in a fresh process.
    Parameters
    ----------
    source : str
        The candidate source code to evaluate.
    dataset : dict[str, dict[str, Any]]
        The dataset to use for evaluation.
    timeout : float, optional
        The maximum time (in seconds) to allow for evaluation, by default 10.0

    Returns
    -------
    dict[str, Any]
        The evaluation results.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    command = [sys.executable, "-m", "src.sandbox", WORKER_FLAG]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps({"source": source, "dataset": dataset}),
            text=True,
            capture_output=True,
            cwd=ROOT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": f"candidate exceeded {timeout:g}s",
        }
    if completed.returncode != 0:
        return {
            "status": "invalid",
            "error": completed.stderr.strip() or "sandbox process failed",
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"status": "invalid", "error": "malformed sandbox response"}


def _load_dataset(split: str) -> dict[str, dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    with (DATA_DIR / f"{split}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


# demo to test sandbox
DEMO_SOURCE = """\
def scoring_fn(item, remaining, capacity):
    gap = remaining - item
    return np.append(-gap, -capacity - 1)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate", type=Path)
    group.add_argument("--demo", action="store_true")
    parser.add_argument("--split", choices=sorted(SPLITS), default="train")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    source = (
        DEMO_SOURCE
        if args.demo
        else args.candidate.read_text(encoding="utf-8")
    )
    result = run_candidate(source, _load_dataset(args.split), args.timeout)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        _worker()
    else:
        main()
