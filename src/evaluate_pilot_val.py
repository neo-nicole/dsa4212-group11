
import argparse
import json
from pathlib import Path

from src.generate_instances import load_dataset
from src.sandbox import run_candidate


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "pilot"
TIMEOUT = 10.0


def evaluate_file(input_path, val_data):
    """Evaluate training-valid candidates on the validation dataset."""

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    output_path = OUTPUT_DIR / f"validation_{input_path.stem}.jsonl"

    evaluated = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as infile, \
            output_path.open("w", encoding="utf-8") as outfile:

        for line in infile:
            if not line.strip():
                continue

            record = json.loads(line)

            # Skip candidates that failed training evaluation.
            if not record.get("valid") or not record.get("candidate_code"):
                skipped += 1
                continue

            try:
                result = run_candidate(
                    source=record["candidate_code"],
                    dataset=val_data,
                    timeout=TIMEOUT,
                )
            except Exception as e:
                result = {
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                }

            validation_record = {
                "method": record["method"],
                "run_id": record["run_id"],
                "generation": record["generation"],
                "training_score": record["evaluation"].get("score"),
                "validation_evaluation": result,
            }

            outfile.write(
                json.dumps(validation_record, allow_nan=False) + "\n"
            )

            evaluated += 1

            print(
                f"{record['method']} | "
                f"Generation {record['generation']} | "
                f"Validation status: {result.get('status')} | "
                f"Validation score: {result.get('score')}"
            )

    print(f"\nInput: {input_path.name}")
    print(f"Candidates evaluated: {evaluated}")
    print(f"Candidates skipped: {skipped}")
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pilot candidates on validation."
    )

    parser.add_argument(
        "candidate_files",
        nargs="+",
        help="Paths to saved pilot candidate JSONL files.",
    )

    args = parser.parse_args()

    val_data = load_dataset("val")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for input_path in args.candidate_files:
        evaluate_file(input_path, val_data)


if __name__ == "__main__":
    main()
