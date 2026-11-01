"""Process-isolated runner for LLM-generated bin-packing heuristics.

Candidates must define:

    def priority_fn(item, remaining, capacity):
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
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation import evaluate, evaluate_instance
from src.generate_instances import load_dataset

ROOT = Path(__file__).resolve().parent.parent
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
    """Compile source containing exactly one priority_fn definition."""
    try:
        tree = ast.parse(source, filename="<candidate>", mode="exec")
    except SyntaxError as exc:
        raise CandidateError(
            f"syntax error: {exc.msg} (line {exc.lineno})"
        ) from exc

    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "priority_fn"
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
            "candidate must contain exactly one priority_fn function"
        )
    function = functions[0]
    if isinstance(function, ast.AsyncFunctionDef):
        raise CandidateError("priority_fn must be a regular function")
    args = function.args
    if (
        len(args.args) != 3
        or args.posonlyargs
        or args.vararg is not None
        or args.kwarg is not None
        or args.kwonlyargs
        or args.defaults
    ):
        raise CandidateError(
            "priority_fn must have exactly three arguments: "
            "(item, remaining, capacity)"
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
    return namespace["priority_fn"]


def _evaluate(
    source: str,
    dataset: dict[str, dict[str, Any]],
    per_instance: bool = False,
) -> dict[str, Any]:
    """Compile a candidate and evaluate a dataset or one instance.

    Dataset mode uses ``evaluate`` for one worker process. Per-instance mode
    uses ``evaluate_instance`` for a worker containing exactly one instance.
    """
    priority_fn = _compile_candidate(source)
    if not per_instance:
        return evaluate(priority_fn, dataset)
    if len(dataset) != 1:
        raise ValueError("per-instance worker requires exactly one instance")
    instance_id, instance = next(iter(dataset.items()))
    result = evaluate_instance(priority_fn, instance_id, instance)
    return {"status": result["status"], "feasible": result["feasible"], "results": [result]}


def _worker() -> None:
    try:
        request = json.load(sys.stdin)
        response = _evaluate(
            request["source"],
            request["dataset"],
            request.get("per_instance", False),
        )
    except (CandidateError, ValueError, KeyError) as exc:
        response = {"status": "invalid", "error": str(exc)}
    except Exception as exc:
        response = {
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(response, allow_nan=False))


def _failure_result(
    instance_id: str,
    instance: dict[str, Any],
    status: str,
    error: str,
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "family": instance.get("family"),
        "num_items": instance.get("num_items"),
        "status": status,
        "feasible": False,
        "bins_used": None,
        "lower_bound": None,
        "score": None,
        "runtime": None,
        "error": error,
    }


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["family"]].append(result)

    family_scores = {
        family: (
            statistics.mean(row["score"] for row in rows)
            if all(row["feasible"] for row in rows)
            else None
        )
        for family, rows in grouped.items()
    }
    feasible = all(row["feasible"] for row in results)
    status = (
        "ok"
        if feasible
        else "timeout"
        if any(row["status"] == "timeout" for row in results)
        else "invalid"
    )
    return {
        "status": status,
        "feasible": feasible,
        "score": (
            statistics.mean(family_scores.values())
            if feasible else None
        ),
        "mean_bins": (
            statistics.mean(row["bins_used"] for row in results)
            if feasible else None
        ),
        "total_runtime": sum(
            row["runtime"] for row in results
            if row["runtime"] is not None
        ),
        "family_scores": family_scores,
        "results": results,
    }


def _run_candidate(
    source: str,
    dataset: dict[str, dict[str, Any]],
    timeout: float = 10.0,
    per_instance: bool = False,
) -> dict[str, Any]:
    """Evaluate a candidate using the selected process isolation strategy.

    By default, one process evaluates the complete dataset. When
    ``per_instance`` is true, each instance gets a fresh process and its own
    timeout.

    This private helper is used by ``run_candidate`` and
    ``run_candidate_per_instance``.
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
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    command = [sys.executable, "-m", "src.sandbox", WORKER_FLAG]

    def run_worker(
        worker_dataset: dict[str, dict[str, Any]],
        worker_per_instance: bool,
    ) -> dict[str, Any] | None:
        try:
            completed = subprocess.run(
                command,
                input=json.dumps({
                    "source": source,
                    "dataset": worker_dataset,
                    "per_instance": worker_per_instance,
                }),
                text=True,
                capture_output=True,
                cwd=ROOT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None

        if completed.returncode != 0:
            return {
                "status": "invalid",
                "error": completed.stderr.strip() or "sandbox process failed",
            }

        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {"status": "invalid", "error": "malformed sandbox response"}
        if not isinstance(response, dict):
            return {"status": "invalid", "error": "malformed sandbox response"}
        return response

    if not per_instance:
        response = run_worker(dataset, False)
        if response is None:
            return {
                "status": "timeout",
                "error": f"candidate exceeded {timeout:g}s",
            }
        return response

    results = []
    for instance_id, instance in dataset.items():
        response = run_worker({instance_id: instance}, True)
        if response is None:
            results.append(
                _failure_result(
                    instance_id,
                    instance,
                    "timeout",
                    f"candidate exceeded {timeout:g}s for {instance_id}",
                )
            )
            continue
        if response.get("status") == "invalid" and "results" not in response:
            results.append(
                _failure_result(
                    instance_id,
                    instance,
                    "invalid",
                    response.get("error", "sandbox process failed"),
                )
            )
            continue
        instance_results = response.get("results")
        if not isinstance(instance_results, list) or len(instance_results) != 1:
            results.append(
                _failure_result(
                    instance_id,
                    instance,
                    "invalid",
                    response.get("error", "malformed instance result"),
                )
            )
            continue
        results.append(instance_results[0])

    if not results:
        raise ValueError("dataset must not be empty")
    return _aggregate_results(results)


def run_candidate(
    source: str,
    dataset: dict[str, dict[str, Any]],
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Evaluate a candidate against a dataset in one fresh process.

    The timeout applies to the complete dataset evaluation.
    """
    return _run_candidate(source, dataset, timeout, per_instance=False)


def run_candidate_per_instance(
    source: str,
    dataset: dict[str, dict[str, Any]],
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Evaluate each dataset instance in a fresh process.

    The timeout applies independently to each complete bin-packing instance.
    """
    return _run_candidate(source, dataset, timeout, per_instance=True)


# demo to test sandbox
DEMO_SOURCE = """\
def priority_fn(item, remaining, capacity):
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
    parser.add_argument(
        "--per-instance",
        action="store_true",
        help="use a fresh worker and timeout for each dataset instance",
    )
    args = parser.parse_args()

    source = (
        DEMO_SOURCE
        if args.demo
        else args.candidate.read_text(encoding="utf-8")
    )
    run = run_candidate_per_instance if args.per_instance else run_candidate
    result = run(source, load_dataset(args.split), args.timeout)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        _worker()
    else:
        main()
