"""Source-based, four-profile evaluation of the entire shared 31-page selection.

Reads saved predictions only. Private source text, per-cell references and detailed
failure traces stay under outputs; public metrics contain counts and hashes.
"""

import argparse
import hashlib
import html
import importlib.util
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import pypdfium2 as pdfium

from multimodal_doc_parser.parser_adapters.table_html import table_cells

ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = (
    ROOT
    / "outputs/quality_evaluations/four-profile-corpus-source-review-01"
    / "source-selected-annotations-v2.json"
)
BLOCK = re.compile(r"<BLOCK>([A-Za-z_][\w-]*)\[([^\]\n]*)\]<FORK>(.*?)(?=<BLOCK>|\Z)", re.S)
LABELS = [
    "PaddleOCR-VL-1.6",
    "dots.mocr Q4_K_M + Q8_0",
    "Unlimited-OCR 2 NF4/FP16",
    "dots.mocr original BF16",
]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(value):
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    parser = Plain()
    parser.feed(value)
    value = " ".join(parser.parts)
    for command, replacement in {
        "times": "x",
        "cdot": "x",
        "text": "",
        "mathrm": "",
        "mathbf": "",
    }.items():
        value = re.sub(r"\\" + command + r"\b", replacement, value)
    # Strip formatting commands, but preserve their braced visible contents.
    value = re.sub(r"\\(?:begin|end)\{[^}]+\}", " ", value)
    value = re.sub(r"\\[A-Za-z]+", " ", value)
    return html.unescape(value)


def tokens(value):
    value = unicodedata.normalize("NFKC", plain(value)).lower()
    value = re.sub(r"(?<=\w)[-\u00ad]\s*\n\s*(?=\w)", "", value)
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    value = value.replace("’", "'").replace("'", "")
    return re.findall(r"[^\W_]+(?:\.\d+)*", value)


def distance(reference, prediction):
    """Exact unit-cost Levenshtein distance using the bit-vector recurrence."""
    if not reference:
        return len(prediction)
    masks = {}
    for index, item in enumerate(reference):
        masks[item] = masks.get(item, 0) | (1 << index)
    limit = (1 << len(reference)) - 1
    high = 1 << (len(reference) - 1)
    positive, negative, score = limit, 0, len(reference)
    for item in prediction:
        equal = masks.get(item, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        score += bool(plus & high) - bool(minus & high)
        plus = (plus << 1) | 1
        minus <<= 1
        positive = (minus | ~(vertical | plus)) & limit
        negative = plus & vertical & limit
    return score


def text_counts(reference, prediction):
    a, b = tokens(reference), tokens(prediction)
    overlap = sum((Counter(a) & Counter(b)).values())
    return {
        "reference_tokens": len(a),
        "prediction_tokens": len(b),
        "matched_tokens": overlap,
        "word_edit_operations": distance(a, b),
    }


def summarize_text(rows):
    totals = {
        key: sum(row[key] for row in rows)
        for key in [
            "reference_tokens",
            "prediction_tokens",
            "matched_tokens",
            "word_edit_operations",
        ]
    }
    r, p, m = (totals[k] for k in ["reference_tokens", "prediction_tokens", "matched_tokens"])
    return {
        **totals,
        "precision": m / p if p else 0,
        "recall": m / r if r else 0,
        "f1": 2 * m / (r + p) if r + p else 0,
        "word_error_rate": totals["word_edit_operations"] / r if r else None,
        "pages": len(rows),
    }


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    aa, bb = max(0, a[2] - a[0]) * max(0, a[3] - a[1]), max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (aa + bb - intersection) if aa + bb > intersection else 0


def maximum_matches(references, predictions, threshold):
    """One prediction cannot earn credit for multiple source regions."""
    visual = {"image", "picture", "figure", "chart", "diagram", "map", "table"}
    edges = []
    for ref in references:
        allowed = {"table"} if ref["category"] == "table" else visual - {"table"}
        if ref["category"] in {"chart", "infographic", "diagram"}:
            allowed = visual
        edges.append(
            [
                i
                for i, p in enumerate(predictions)
                if p["label"] in allowed
                and p.get("bbox")
                and iou(ref["bbox"], p["bbox"]) >= threshold
            ]
        )
    owners = {}

    def augment(index, visited):
        for candidate in edges[index]:
            if candidate in visited:
                continue
            visited.add(candidate)
            if candidate not in owners or augment(owners[candidate], visited):
                owners[candidate] = index
                return True
        return False

    for index in range(len(references)):
        augment(index, set())
    return set(owners.values())


def normalized_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    values = [float(x) / s for x, s in zip(box, [width, height] * 2, strict=True)]
    return values if values[0] < values[2] and values[1] < values[3] else None


def load_page(run, number, profile):
    raw = run / ("raw/paddleocr" if profile == 0 else "raw/dots_mocr")
    path = run / f"page_{number:04d}.json" if profile == 2 else raw / f"page_{number:04d}.json"
    if not path.exists():
        path = raw / f"response_{number:04d}.json"
    if not path.exists():
        return {"complete": False, "text": "", "blocks": [], "artifact": None}
    data = read(path)
    blocks = []
    complete = data.get("finish_reason", "stop") == "stop"
    if profile == 0:
        data = data.get("res", data)
        width, height = data["width"], data["height"]
        for item in data.get("parsing_res_list", []):
            blocks.append(
                {
                    "label": item["block_label"].lower(),
                    "text": item.get("block_content", ""),
                    "bbox": normalized_box(item.get("block_bbox"), width, height),
                }
            )
    elif profile == 2:
        for match in BLOCK.finditer(data.get("text", "")):
            try:
                box = json.loads("[" + match[2] + "]")
            except (ValueError, TypeError):
                box = None
            blocks.append(
                {
                    "label": match[1].lower(),
                    "text": match[3].replace("<PAGE>", ""),
                    "bbox": normalized_box(box, 1000, 1000),
                }
            )
    else:
        try:
            items = json.loads(data.get("content", ""))
            if not isinstance(items, list):
                raise ValueError("Non-list layout")
            for item in items:
                blocks.append(
                    {
                        "label": str(item.get("category", "missing")).lower(),
                        "text": item.get("text", ""),
                        "bbox": normalized_box(
                            item.get("bbox"), data["input_width"], data["input_height"]
                        ),
                    }
                )
        except (ValueError, TypeError, KeyError):
            complete = False
    text = "\n".join(b["text"] for b in blocks) if complete else ""
    return {
        "complete": complete,
        "text": text,
        "blocks": blocks if complete else [],
        "artifact": str(path.relative_to(ROOT)),
        "artifact_sha256": digest(path),
    }


def html_tables(page):
    candidates = []
    for block in page["blocks"]:
        for value in re.findall(r"<table\b.*?</table>", block["text"], re.I | re.S):
            candidates.append(value)
    return candidates


def parsed_tables(page):
    result = []
    for value in html_tables(page):
        try:
            result.append(table_cells(value)[2])
        except (ValueError, TypeError):
            pass
    return result


def check_nasa_table(page, reference):
    rows = []
    groups = []
    tables = html_tables(page)
    # Read explicit HTML row boundaries, even if malformed rowspan occupancy fails.
    raw_rows = [
        tokens(row)
        for table in tables
        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.I | re.S)
    ]
    for objective, values in reference["values"].items():
        matching = [r for r in raw_rows if objective in r]
        for index, value in enumerate(values):
            ok = any(
                len(row) >= 3 and row[-2 + index if index == 0 else -1] == str(value)
                for row in matching
            )
            rows.append({"objective": objective, "column": index, "passed": ok})
    # A valid table parser must also accept the span geometry.
    for name, span in reference["group_rowspans"].items():
        ok = any(
            tokens(cell.text)[:1] == [name.lower()] and cell.row_span == span
            for cells in parsed_tables(page)
            for cell in cells
        )
        groups.append({"group": name, "passed": ok})
    return rows, groups


def sensor_checks(page, reference):
    tables = parsed_tables(page)
    result = []
    for ref in reference["table_cells"]:
        cells = tables[ref["table_index"]] if ref["table_index"] < len(tables) else []
        actual = next((c for c in cells if c.row == ref["row"] and c.column == ref["column"]), None)
        ok = (
            actual is not None
            and tokens(actual.text) == tokens(ref["text"])
            and actual.row_span == ref["row_span"]
            and actual.column_span == ref["column_span"]
        )
        result.append({"row": ref["row"], "column": ref["column"], "passed": ok})
    return result


def chart_checks(page, references):
    results = []
    for ref in references:
        # Source-located evidence only: same values in the page's sensor table do not count.
        blocks = [
            b
            for b in page["blocks"]
            if b.get("bbox")
            and iou(b["bbox"], ref["region"]) >= 0.1
            and (b["bbox"][1] + b["bbox"][3]) / 2 > ref["region"][1]
            and (b["bbox"][1] + b["bbox"][3]) / 2 < ref["region"][3]
        ]
        candidate_rows = []
        for block in blocks:
            candidate_rows.extend(re.findall(r"<tr\b[^>]*>(.*?)</tr>", block["text"], re.I | re.S))
            candidate_rows.extend(block["text"].splitlines())
        expected = tokens(ref["category"]) + tokens(ref["value"])
        ok = any(all(v in tokens(row) for v in expected) for row in candidate_rows)
        results.append({"id": ref["id"], "passed": ok})
    return results


def rate(checks):
    return {"passed": sum(c["passed"] for c in checks), "checked": len(checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--public-metrics",
        type=Path,
        default=ROOT / "evaluation/results/parser-corpus-four-profiles-v1.json",
    )
    args = parser.parse_args()
    destination = args.output_directory.resolve()
    if destination.exists():
        parser.error("Choose a fresh output directory; evaluation evidence is immutable")
    destination.mkdir(parents=True)
    refs = read(args.annotations)
    manifest = read(ROOT / "evaluation/four-parser-profiles-showcase-cases.json")
    cases = [c for c in manifest["cases"] if c["group"] == "parsers"]
    # Use the exact upstream-prompt Q4 job whose sent PNGs fed the BF16 cloud run.
    original = next(c for c in cases if c["id"] == "original-private")
    original["runs"][1]["directory"] = "dots-mocr-q4-k-m-local-validation-02-upstream-prompt"
    input_manifest = read(
        ROOT
        / "outputs/parser_runs/dots-mocr-original-hf-bf16-colab-31-pages-01"
        / "remote/input-manifest.json"
    )
    source_ids = [c["source_document_id"] for c in input_manifest["cases"]]
    frozen, page_data = {}, {}
    source_inventory = []
    for case in cases:
        cloud_run = ROOT / "outputs/parser_runs" / case["runs"][3]["directory"]
        source = cloud_run / "source.pdf"
        source_id = "sha256:" + digest(source)
        if source_id not in source_ids:
            raise ValueError("Source PDF differs from matched cloud selection")
        with pdfium.PdfDocument(source) as pdf:
            if len(pdf) != len(case["original_pages"]):
                raise ValueError("Page selection mismatch")
            frozen[case["id"]] = [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
        source_inventory.append(
            {
                "case": case["id"],
                "source_sha256": source_id,
                "pages": len(case["original_pages"]),
                "original_pages": case["original_pages"],
            }
        )
        for profile, spec in enumerate(case["runs"]):
            run = ROOT / "outputs/parser_runs" / spec["directory"]
            document = (
                read(run / "document.json")
                if (run / "document.json").exists()
                else read(run / "run-status.json")
            )
            if document["document_id"] != source_id:
                raise ValueError("Model prediction source identity mismatch")
            for number in range(1, len(case["original_pages"]) + 1):
                page_data[case["id"], profile, number] = load_page(run, number, profile)
    save(destination / "frozen-native-text-reference.json", frozen)
    save(destination / "selected-source-annotations.json", refs)
    math_source = ROOT / "scripts/parser_evaluation/evaluate_handwritten_math_comparison.py"
    spec = importlib.util.spec_from_file_location("frozen_math_reference_scorer", math_source)
    math_scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_scorer)
    math_scorer.RUNS[LABELS[3]] = "dots-mocr-bf16-colab-author-mathematics-4-pages-01"
    math = math_scorer.evaluate(math_scorer.REFERENCE)
    profiles = []
    traces = []
    for profile, label in enumerate(LABELS):
        printed, strict_printed = [], []
        case_summaries = []
        complete_total, accepted_total = 0, 0
        for case in cases:
            run = ROOT / "outputs/parser_runs" / case["runs"][profile]["directory"]
            canonical = (
                read(
                    run
                    / "raw"
                    / ("paddleocr" if profile == 0 else "dots_mocr")
                    / "inference-manifest.json"
                )
                if profile != 2
                else {}
            )
            accepted = (
                canonical.get("completed_pages", len(canonical.get("pages", [])))
                if profile != 2
                else None
            )
            completed = sum(
                page_data[case["id"], profile, n]["complete"]
                for n in range(1, len(case["original_pages"]) + 1)
            )
            complete_total += completed
            accepted_total += accepted or 0
            rows = []
            accepted_numbers = {p["page_number"] for p in canonical.get("pages", [])}
            for number, source_text in enumerate(frozen[case["id"]], 1):
                page = page_data[case["id"], profile, number]
                trace = {
                    "case": case["id"],
                    "profile": label,
                    "page": number,
                    "original_page": case["original_pages"][number - 1],
                    **{k: v for k, v in page.items() if k not in {"blocks", "text"}},
                }
                if source_text.strip():
                    counts = text_counts(source_text, page["text"])
                    rows.append(counts)
                    printed.append(counts)
                    # Experimental profile has no supported canonical adapter.
                    if profile != 2:
                        strict_printed.append(
                            text_counts(
                                source_text, page["text"] if number in accepted_numbers else ""
                            )
                        )
                    trace["printed_text"] = counts
                    trace["scored_text_sha256"] = hashlib.sha256(page["text"].encode()).hexdigest()
                traces.append(trace)
            case_summaries.append(
                {
                    "case": case["id"],
                    "requested": len(case["original_pages"]),
                    "finished_generations": completed,
                    "canonical_accepted": accepted,
                    "printed_text": summarize_text(rows) if rows else None,
                    "run_directory": str(run.relative_to(ROOT)),
                }
            )
        scanned = []
        for ref in refs["scanned_spans"]:
            page = page_data[ref["case"], profile, ref["page"]]
            expected = " ".join(tokens(ref["expected"]))
            actual = " ".join(tokens(page["text"]))
            scanned.append(
                {
                    "id": ref["id"],
                    "category": ref["category"],
                    "passed": bool(expected) and expected in actual,
                }
            )
        regions = []
        for threshold in [0.25, 0.5]:
            grouped = defaultdict(list)
            for ref in refs["regions"]:
                grouped[ref["case"], ref["page"]].append(ref)
            for (case, number), references in grouped.items():
                page = page_data[case, profile, number]
                found = maximum_matches(references, page["blocks"], threshold)
                regions.extend(
                    {
                        "id": ref["id"],
                        "category": ref["category"],
                        "threshold": threshold,
                        "passed": i in found,
                    }
                    for i, ref in enumerate(references)
                )
        nasa, spans = check_nasa_table(
            page_data["nasa-financial", profile, 17], refs["nasa_budget"]
        )
        sensor = sensor_checks(page_data["original-private", profile, 1], refs["sensor_table"])
        chart = chart_checks(page_data["original-private", profile, 1], refs["chart_checks"])
        m = math["results"][profile]
        save(
            destination / f"profile-{profile + 1}-detailed-checks.json",
            {
                "scanned": scanned,
                "regions": regions,
                "nasa_table": nasa,
                "nasa_spans": spans,
                "sensor_table": sensor,
                "chart_values": chart,
            },
        )
        profile_result = {
            "profile": label,
            "requested_pages": 31,
            "finished_generations": complete_total,
            "canonical_accepted": accepted_total if profile != 2 else None,
            "cases": case_summaries,
            "printed_text": summarize_text(printed),
            "strict_pipeline_printed_text": summarize_text(strict_printed)
            if strict_printed
            else None,
            "diary_printed_spans": rate([c for c in scanned if c["category"] == "diary_printed"]),
            "diary_handwriting_spans": rate(
                [c for c in scanned if c["category"] == "diary_handwriting"]
            ),
            "assignment_assertions": m["selected_assertions"],
            "assignment_categories": m["categories"],
            "assignment_table_cells": {
                "passed": m["truth_table"]["passed"],
                "checked": m["truth_table"]["checked"],
            },
            "nasa_budget_cells": rate(nasa),
            "nasa_goal_rowspans": rate(spans),
            "sensor_cells_and_spans": rate(sensor),
            "chart_category_value_pairs": rate(chart),
            "visual_localization": {
                category: {
                    str(t): rate(
                        [c for c in regions if c["category"] == category and c["threshold"] == t]
                    )
                    for t in [0.25, 0.5]
                }
                for category in sorted({c["category"] for c in regions})
            },
        }
        profile_result["printed_document_macro_f1"] = statistics.mean(
            c["printed_text"]["f1"] for c in case_summaries if c["printed_text"]
        )
        profiles.append(profile_result)
        print(
            label,
            "text F1",
            round(profile_result["printed_text"]["f1"], 4),
            "NASA cells",
            rate(nasa),
            "rowspans",
            rate(spans),
            "diary",
            rate(scanned),
            flush=True,
        )
    save(destination / "per-page-traces.json", traces)
    report = {
        "version": "1.0",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "All 31 shared pages; complete PDF-native text on 22 pages plus "
            "selected source checks on scans and structures"
        ),
        "source_inventory": source_inventory,
        "reference_status": (
            "Retrospective assistant source review plus PDF-native text; "
            "not independent human labels or held-out benchmark"
        ),
        "source_annotations_sha256": digest(args.annotations),
        "native_text_reference_sha256": digest(destination / "frozen-native-text-reference.json"),
        "math_reference_sha256": math["reference_sha256"],
        "math_evaluator_sha256": math["evaluator_sha256"],
        "evaluator_sha256": digest(Path(__file__)),
        "reference_exclusions": refs["excluded_scanned_content"],
        "profiles": profiles,
        "page_results": traces,
        "limitations": [
            "IRENA/IMF lack outputs from all four profiles and are excluded from ranking.",
            (
                "Printed token F1 measures content overlap including repetitions, not "
                "semantic correctness, layout or reading order."
            ),
            (
                "WER uses native PDF sequence; differences include source text-layer order "
                "and omitted raster-only text."
            ),
            (
                "Missing, timed-out and unparseable generations contribute empty text and "
                "failed checks; no completed-page-only accuracy."
            ),
            (
                "Raw content of schema-invalid pages is scored separately from strict "
                "canonical pipeline yield."
            ),
            "Scanned pages have selected assertions, not complete CER/WER references.",
            (
                "Visual checks measure localization recall at IoU 0.5 with 0.25 sensitivity; "
                "partial region annotations cannot establish precision or semantic "
                "description quality."
            ),
            "Only three bar category/value pairs are available; no general chart-model ranking.",
            (
                "Assignment scores reuse unchanged selected reference/evaluator; no model "
                "answers repaired."
            ),
            (
                "Different hardware, image preprocessing, runtime backends and conversion "
                "lineage prevent causal quantization and speedup claims."
            ),
            (
                "Original Q4 comparison uses the upstream-prompt job matching BF16 sent "
                "PNGs, rather than the later project-prompt rerun."
            ),
        ],
    }
    save(args.public_metrics, report)
    save(destination / "metrics.json", report)
    files = {
        str(p.relative_to(destination)): digest(p) for p in destination.rglob("*") if p.is_file()
    }
    save(destination / "file-checksums.json", files)
    print("Saved", args.public_metrics, flush=True)


if __name__ == "__main__":
    main()
