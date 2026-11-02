"""Package curated public excerpts from the existing saved-results explorer."""

import argparse
from pathlib import Path

from multimodal_doc_parser.evaluation.public_demo import build_public_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showcase-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path, default=Path("evaluation/public-demo-selection.json")
    )
    args = parser.parse_args()
    print(
        build_public_demo(
            args.showcase_directory,
            args.output_directory,
            args.selection,
            evidence_directory=Path("evaluation/results"),
        )
    )


if __name__ == "__main__":
    main()
