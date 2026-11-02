"""Score frozen selected source assertions and ordered truth-table cells.

This is a small assistant-reviewed development evaluation, not full-document
accuracy. All requested pages stay in the denominator, including failed jobs.
"""

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = (
    ROOT / "evaluation/reference_annotations/author-handwritten-mathematics-selected-v1.json"
)
RUNS = {
    "PaddleOCR-VL-1.6": "paddleocr-author-handwritten-mathematics-four-pages-01",
    "dots.mocr Q4_K_M + Q8_0 projector": (
        "dots-mocr-q4-k-m-author-handwritten-mathematics-four-pages-01"
    ),
    "Unlimited-OCR 2 NF4 / FP16 vision": (
        "unlimited-ocr-2-nf4-author-handwritten-mathematics-four-pages-01"
    ),
}


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def normalize(value: str) -> str:
    """Fold typography and explicit TeX commands, retaining mathematical operators."""
    value = unicodedata.normalize("NFKC", unescape(value))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\\(?:mathbb|mathrm|mathbf|text|operatorname)\s*\{([^{}]*)\}", r" \1 ", value)
    value = re.sub(r"([_^])\{([^{}]*)\}", r"\1\2", value)
    value = value.replace(r"\{", "{").replace(r"\}", "}")
    for command, symbol in {
        "setminus": "\\",
        "backslash": "\\",
        "cup": "∪",
        "cap": "∩",
        "in": "∈",
        "leq": "≤",
        "le": "≤",
        "times": "×",
        "cdot": "×",
        "circ": "∘",
        "mapsto": "↦",
        "to": "→",
        "rightarrow": "→",
        "neg": "¬",
        "lnot": "¬",
        "land": "∧",
        "wedge": "∧",
        "lor": "∨",
        "vee": "∨",
        "mid": "|",
        "vert": "|",
    }.items():
        value = re.sub(r"\\" + command + r"(?![A-Za-z])", lambda _, symbol=symbol: symbol, value)
    value = re.sub(r"\\(?:left|right|mathbb|mathrm|mathbf|text|operatorname)\b", "", value)
    value = value.replace("’", "'").replace("−", "-").replace("·", "×")
    value = value.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    return "".join(c.lower() for c in value if c.isalnum() or c in "=+^-≤≥<>∪∩∈\\×∘↦→¬∧∨|(){}")


class Tables(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self.table: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append("".join(self.cell))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def tables(text: str) -> list[list[list[str]]]:
    parser = Tables()
    parser.feed(text)
    markdown: list[list[str]] = []
    for line in text.splitlines() + [""]:
        if line.strip().startswith("|"):
            row = line.strip().strip("|").split("|")
            if not all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in row):
                markdown.append(row)
        elif markdown:
            parser.tables.append(markdown)
            markdown = []
    return parser.tables


def page_text(run: Path, number: int, profile: str) -> tuple[str, str]:
    if profile.startswith("Unlimited"):
        data = read(run / f"page_{number:04d}.json")
        content = re.sub(
            r"<BLOCK>[A-Za-z_][\w-]*\[[^\]\n]*\]<FORK>", "\n", data.get("text", "")
        ).replace("<PAGE>", "")
        return content, "experimental raw block contents; layout envelopes stripped"
    if profile.startswith("Paddle"):
        data = read(run / f"raw/paddleocr/page_{number:04d}.json")
        data = data.get("res", data)
        return "\n".join(
            str(b.get("block_content", "")) for b in data.get("parsing_res_list", [])
        ), "raw block content"
    path = run / f"raw/dots_mocr/page_{number:04d}.json"
    if not path.exists():
        path = run / f"page-jobs/page_{number:04d}/run/raw/dots_mocr/page_0001.json"
    content = read(path).get("content", "")
    try:
        blocks = json.loads(content)
        if not isinstance(blocks, list):
            return "", "invalid layout JSON"
        return "\n".join(str(b.get("text", "")) for b in blocks), "raw layout block text"
    except (json.JSONDecodeError, TypeError):
        return "", "unparseable/absent raw layout JSON; zero recovered assertions"


def table_score(text: str, reference: dict) -> dict:
    """Evaluate a table only with the expected ordered header; no best-window search."""
    expected_headers = [normalize(s) for s in reference["headers"]]
    candidates = tables(text)
    matched = next(
        (t for t in candidates if t and [normalize(s) for s in t[0]] == expected_headers), None
    )
    expected = reference["rows"]
    body = matched[1:] if matched else []
    cells = []
    for row_number, row in enumerate(expected):
        for column, value in enumerate(row):
            actual = (
                body[row_number][column]
                if row_number < len(body) and column < len(body[row_number])
                else None
            )
            cells.append(
                {
                    "row": row_number + 1,
                    "column": column + 1,
                    "expected": value,
                    "actual": actual,
                    "passed": actual is not None and normalize(actual) == normalize(value),
                }
            )
    return {
        "checked": 20,
        "passed": sum(c["passed"] for c in cells),
        "ordered_header_matched": matched is not None,
        "detected_tables": len(candidates),
        "exact_shape": len(body) == 4 and all(len(row) == 5 for row in body),
        "cells": cells,
    }


def evaluate(reference_path: Path) -> dict:
    reference = read(reference_path)
    selection = read(ROOT / reference["source_selection"])
    results = []
    for profile, directory in RUNS.items():
        run = ROOT / "outputs/parser_runs" / directory
        status = read(run / "run-status.json")
        page_outputs = {n: page_text(run, n, profile) for n in range(1, 5)}
        checks = []
        for item in reference["assertions"]:
            text, origin = page_outputs[item["page"]]
            normalized = normalize(text)
            variants = [item["expected"], *item["accepted_format_variants"]]
            if item["category"] == "text":
                # Prose punctuation may wrap a word without changing its transcription.
                normalized = normalized.translate(str.maketrans("", "", "(){}"))
                matched = next(
                    (
                        s
                        for s in variants
                        if normalize(s).translate(str.maketrans("", "", "(){}")) in normalized
                    ),
                    None,
                )
            else:
                matched = next((s for s in variants if normalize(s) in normalized), None)
            checks.append(
                {
                    **item,
                    "passed": matched is not None,
                    "matched_variant": matched,
                    "output_origin": origin,
                }
            )
        raw_dir = "paddleocr" if profile.startswith("Paddle") else "dots_mocr"
        manifest = read(run / "raw" / raw_dir / "inference-manifest.json")
        canonical = read(run / "document.json")
        source_id = manifest.get(
            "document_id", status.get("document_id", canonical.get("document_id"))
        )
        if source_id is not None and source_id != "sha256:" + selection["source_sha256"]:
            raise ValueError(f"Model run has a different source document: {directory}")
        diagram_regions = [
            e
            for e in canonical.get("elements", [])
            if e["type"] == "image"
            and any(r["page_number"] == 2 for r in e.get("source_regions", []))
        ]
        timing_rows = manifest.get("pages", status.get("pages", []))
        durations = [p.get("elapsed_seconds", p.get("attempt_seconds")) for p in timing_rows]
        durations = [d for d in durations if d is not None]
        metrics = {
            category: {
                "passed": sum(c["passed"] for c in checks if c["category"] == category),
                "checked": sum(c["category"] == category for c in checks),
            }
            for category in ("text", "math")
        }
        results.append(
            {
                "model": profile,
                "run_directory": str(run.relative_to(ROOT)),
                "run_status": status.get("status", "not_started"),
                "requested_pages": 4,
                "completed_pages": status.get(
                    "completed_pages", manifest.get("completed_pages", 0)
                ),
                "selected_assertions": {
                    "passed": sum(c["passed"] for c in checks),
                    "checked": len(checks),
                },
                "categories": metrics,
                "diagram_diagnostics": {
                    "page": 2,
                    "canonical_model_image_regions": sum(
                        e["provenance"]["producer"] != "PDF native image recovery"
                        for e in diagram_regions
                    )
                    if canonical
                    else None,
                    "experimental_raw_image_regions": (
                        len(
                            re.findall(
                                r"<BLOCK>image\[", read(run / "page_0002.json").get("text", "")
                            )
                        )
                        if profile.startswith("Unlimited")
                        else None
                    ),
                    "scope": (
                        "Image-region count only; not diagram semantic accuracy or recall. "
                        "Native full-photo recovery is not credited as model recognition."
                    ),
                },
                "truth_table": table_score(page_outputs[4][0], reference["truth_table"]),
                "page_seconds": durations,
                "initialization_seconds": manifest.get("initialization_seconds"),
                "page_load_seconds": [p.get("load_seconds") for p in status.get("pages", [])],
                "page_attempt_seconds_including_load": [
                    p.get("attempt_seconds_including_load") for p in status.get("pages", [])
                ],
                "page_output_diagnostics": [
                    {
                        "page": number,
                        "characters": len(text),
                        "scored_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "max_repeated_16_token_sequence_count": max(
                            Counter(
                                tuple(tokens[offset : offset + 16])
                                for offset in range(max(0, len(tokens) - 15))
                            ).values(),
                            default=0,
                        ),
                    }
                    for number, (text, _) in page_outputs.items()
                    for tokens in [re.findall(r"\w+|[^\s\w]", text)]
                ],
                "median_page_seconds": statistics.median(durations) if durations else None,
                "timing_scope": (
                    "Runtime page timings; model loading excluded; "
                    "missing/failed page timings may be absent"
                ),
                "checks": checks,
            }
        )
    return {
        "version": "1.0",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": selection["source_sha256"],
        "reference": str(reference_path.relative_to(ROOT)),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "annotation_status": reference["annotation_status"],
        "scope": reference["scope"],
        "limitations": reference["rules"],
        "gpu": "NVIDIA GeForce GTX 1660 Ti, 6144 MiB",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation/results/author-handwritten-mathematics-comparison-v1.json",
    )
    args = parser.parse_args()
    result = evaluate(args.reference)
    terminal = {"completed", "completed_with_errors", "failed", "interrupted"}
    if not args.allow_partial and any(r["run_status"] not in terminal for r in result["results"]):
        parser.error(
            "A model run is unfinished; use --allow-partial for a provisional local report"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for row in result["results"]:
        print(
            row["model"],
            row["run_status"],
            row["selected_assertions"],
            row["truth_table"]["passed"],
        )


if __name__ == "__main__":
    main()
