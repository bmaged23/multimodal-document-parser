"""Reproduce source-reviewed quality checks against a saved parser run; no inference."""

import argparse
import json
from pathlib import Path

from multimodal_doc_parser.evaluation.source_review import evaluate_source_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new results file; do not overwrite an earlier evaluation")
    result = evaluate_source_review(args.run, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
