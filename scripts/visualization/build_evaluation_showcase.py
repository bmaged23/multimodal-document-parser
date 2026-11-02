"""Build the read-only parser/pipeline showcase from saved runs, without inference."""

import argparse
from pathlib import Path

from multimodal_doc_parser.evaluation.showcase import build_showcase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/showcase-cases.json"))
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include the original private regression sample in this local build",
    )
    args = parser.parse_args()
    print(
        build_showcase(
            Path.cwd(), args.output_directory, args.manifest, include_private=args.include_private
        )
    )


if __name__ == "__main__":
    main()
