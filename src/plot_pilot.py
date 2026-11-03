import json
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "outputs" / "pilot"
OUTPUT_FILE = PILOT_DIR / "pilot_curves.png"


def load_candidates(path):
    candidates = []

    with open(path, "r") as file:
        for line in file:
            if line.strip():
                candidates.append(json.loads(line))

    return candidates


def get_curve(candidates):
    # Keep only valid candidates with an evaluation score.
    valid = [
        c for c in candidates
        if c.get("valid")
        and c.get("evaluation")
        and c["evaluation"].get("score") is not None
    ]

    # Sort candidates by generation number.
    valid.sort(key=lambda c: c.get("generation", 0))

    if not valid:
        return [], []

    tokens = []
    best_scores = []

    best_so_far = float("inf")

    for candidate in valid:
        x = candidate.get("cumulative_tokens", 0)
        score = candidate["evaluation"]["score"]

        best_so_far = min(best_so_far, score)

        tokens.append(x)
        best_scores.append(best_so_far)

    return tokens, best_scores


def get_actual_curve(candidates, window=20):
    # Keep valid generated candidates (exclude generation 0).
    valid = [
        c for c in candidates
        if c.get("generation", 0) > 0
        and c.get("valid")
        and c.get("evaluation")
        and c["evaluation"].get("score") is not None
    ]

    # Sort by generation number.
    valid.sort(key=lambda c: c.get("generation", 0))

    tokens = [
        c.get("cumulative_tokens", 0)
        for c in valid
    ]

    scores = [
        c["evaluation"]["score"]
        for c in valid
    ]

    # Calculate rolling average over 20 valid candidates.
    rolling_scores = []
    rolling_tokens = []

    for i in range(window - 1, len(scores)):
        average = sum(scores[i - window + 1:i + 1]) / window
        rolling_scores.append(average)
        rolling_tokens.append(tokens[i])

    return tokens, scores, rolling_tokens, rolling_scores

def main():
    files = {
        "Method A": sorted(PILOT_DIR.glob("candidates_pilot_A_*.jsonl")),
        "Method B": sorted(PILOT_DIR.glob("candidates_pilot_B_*.jsonl")),
    }

    for method, paths in files.items():
        if len(paths) != 1:
            raise RuntimeError(
                f"Expected exactly one {method} pilot file, found {len(paths)}."
            )

    # Load both methods' candidate records.
    data = {
        method: load_candidates(paths[0])
        for method, paths in files.items()
    }

    # GRAPH 1: Actual candidate scores + rolling average.
    plt.figure(figsize=(10, 6))

    for method, candidates in data.items():
        tokens, scores, rolling_tokens, rolling_scores = (
            get_actual_curve(candidates)
        )

        plt.scatter(
            tokens,
            scores,
            alpha=0.25,
            s=15,
            label=f"{method} actual scores",
        )

        plt.plot(
            rolling_tokens,
            rolling_scores,
            linewidth=2,
            label=f"{method} 20-candidate rolling average",
        )

    plt.xlabel("Cumulative tokens used so far")
    plt.ylabel("% Excess over lower bound")
    plt.title("Pilot: Actual Candidate Scores vs Token Usage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    actual_output = PILOT_DIR / "pilot_curves.png"
    plt.savefig(actual_output, dpi=300)
    plt.close()

    print(f"Actual candidate score plot saved to: {actual_output}")

    # GRAPH 2: Original best-so-far curves.
    plt.figure(figsize=(10, 6))

    for method, candidates in data.items():
        tokens, best_scores = get_curve(candidates)

        if best_scores:
            plt.step(
                tokens,
                best_scores,
                where="post",
                linewidth=2,
                label=method,
            )

    plt.xlabel("Cumulative tokens used so far")
    plt.ylabel("% Excess over lower bound (best-so-far)")
    plt.title("Pilot: Best-so-Far Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    best_output = PILOT_DIR / "best_so_far_curves.png"
    plt.savefig(best_output, dpi=300)
    plt.close()

    print(f"Best-so-far plot saved to: {best_output}")


if __name__ == "__main__":
    main()
