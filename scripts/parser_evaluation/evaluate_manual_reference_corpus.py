"""Score frozen assistant source transcriptions against saved native OCR blocks only.

References are assistant-created and non-blind; results are development measurements.
No PDF text extraction, model inference, visual enrichment, or prediction repair is used.
"""

import argparse
import html
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

from multimodal_doc_parser.evaluation.reference_review import digest, save_atomic, validate_value
from multimodal_doc_parser.parser_adapters.table_html import table_cells

ROOT = Path(__file__).resolve().parents[2]
VISUAL = {"picture", "image", "figure", "chart", "diagram", "map"}


class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(value):
    value = str(value)
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    # Avoid interpreting handwritten inequalities as HTML tags.
    value = re.sub(r"<(?!/?[A-Za-z][\w-]*(?:\s|/?>))", "&lt;", value)
    parser = Plain()
    parser.feed(value)
    return html.unescape(" ".join(parser.parts))


def normalized(value):
    value = unicodedata.normalize("NFKC", plain(value)).replace("\u00ad", "")
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "−": "-",
        "–": "-",
        "—": "-",
        "\\%": "%",
        "\\&": "&",
    }
    for a, b in replacements.items():
        value = value.replace(a, b)
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)
    # Formatting-only LaTeX commands retain their contents. Mathematical commands stay visible.
    value = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    value = re.sub(r"(?m)^\s*#{1,6}\s+", "", value)
    value = value.replace("●", "•")
    value = re.sub(r"\.(?:\s*\.){2}", "...", value)
    return re.sub(r"\s+", " ", value).strip()


def edits(reference, prediction, substring=False):
    """Exact bit-vector Levenshtein distance for strings or word sequences."""
    if not reference:
        return len(prediction)
    masks = {}
    for i, item in enumerate(reference):
        masks[item] = masks.get(item, 0) | (1 << i)
    limit = (1 << len(reference)) - 1
    high = 1 << (len(reference) - 1)
    positive, negative, score = limit, 0, len(reference)
    best = score
    for item in prediction:
        equal = masks.get(item, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        score += bool(plus & high) - bool(minus & high)
        plus = (plus << 1) | (0 if substring else 1)
        minus <<= 1
        positive = (minus | ~(vertical | plus)) & limit
        negative = plus & vertical & limit
        best = min(best, score)
    return best if substring else score


def overlap(a, b):
    if not a or not b:
        return 0.0, 0.0
    inter = max(0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])) * max(
        0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
    )
    aa = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
    bb = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    return inter / (aa + bb - inter) if aa + bb > inter else 0.0, inter / min(aa, bb) if min(
        aa, bb
    ) > 0 else 0.0


def minimum_assignment(cost):
    """Rectangular Hungarian assignment with zero-cost dummy columns."""
    n = len(cost)
    m = max(n, len(cost[0]))
    matrix = [row + [0.0] * (m - len(row)) for row in cost]
    u, v, owner, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        slack, used = [float("inf")] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = owner[j0], float("inf"), 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < slack[j]:
                        slack[j], way[j] = cur, j0
                    if slack[j] < delta:
                        delta, j1 = slack[j], j
            for j in range(m + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    slack[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while True:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
            if not j0:
                break
    pairs = [(owner[j] - 1, j - 1) for j in range(1, m + 1) if owner[j] and j <= len(cost[0])]
    return [r for r, _ in pairs], [p for _, p in pairs]


def assign(refs, preds, threshold):
    if not refs or not preds:
        return {}
    weights = [[-overlap(r["source_box"], p["source_box"])[0] for p in preds] for r in refs]
    rr, pp = minimum_assignment(weights)
    return {int(r): int(p) for r, p in zip(rr, pp, strict=True) if -weights[r][p] >= threshold}


def keyed_tail(row, key):
    """Read a keyed row, including an objective ID joined to its description."""
    key = normalized(key)
    matches = []
    for pc in row:
        value = normalized(pc.text)
        tail = [
            normalized(c.text) for c in sorted(row, key=lambda c: c.column) if c.column > pc.column
        ]
        if value == key:
            matches.append(tail)
        elif re.fullmatch(r"\d+\.\d+", key) and value.startswith(key + " "):
            matches.append([value[len(key) + 1 :], *tail])
    return matches


def count_match(expected, actual):
    a, b = Counter(expected), Counter(actual)
    return dict(
        reference=sum(a.values()), prediction=sum(b.values()), matched=sum((a & b).values())
    )


def ratio(n, d):
    return n / d if d else None


def summarize(records):
    # Each family retains its own denominator. Unavailable measurements remain null.
    out = {}
    for category in ["text", "table", "visual", "chart", "heading", "order"]:
        rows = [r[category] for r in records if category in r]
        total = Counter()
        for row in rows:
            for k, v in row.items():
                if type(v) in {int, float}:
                    total[k] += v
        result = dict(total)
        if category == "text":
            result["character_error_rate"] = ratio(
                total["character_edits"], total["reference_characters"]
            )
            result["word_error_rate"] = ratio(total["word_edits"], total["reference_words"])
            result["token_f1"] = ratio(
                2 * total["token_matched"], total["token_reference"] + total["token_prediction"]
            )
        elif category == "table":
            result["strict_cell_text_match"] = ratio(total["exact_cells"], total["reference_cells"])
            result["cell_text_multiset_f1"] = ratio(
                2 * total["content_matched"],
                total["content_reference"] + total["content_prediction"],
            )
            result["keyed_row_value_match"] = ratio(
                total["keyed_row_matches"], total["reference_keyed_rows"]
            )
            result["span_match"] = ratio(total["span_matches"], total["span_attributes"])
            result["header_role_match"] = ratio(total["header_matches"], total["reference_cells"])
        elif category == "visual":
            result["region_recall_at_iou_0_5"] = ratio(
                total["matched_regions"], total["reference_regions"]
            )
            result["matched_mean_iou"] = ratio(total["iou_sum"], total["matched_regions"])
        elif category == "chart":
            result["printed_numeric_token_recall"] = ratio(
                total["matched_values"], total["reference_values"]
            )
        elif category == "heading":
            result["heading_detection_recall"] = ratio(
                total["matched_headings"], total["reference_headings"]
            )
        elif category == "order":
            result["matched_block_pair_order"] = ratio(
                total["correct_pairs"], total["eligible_pairs"]
            )
        out[category] = result
    out["page_slots"] = len(records)
    out["native_present"] = sum(r["native_present"] for r in records)
    out["structured_native_pages"] = sum(r["native_blocks"] > 0 for r in records)
    # Macro rates give each scorable page equal weight; micro rates above weight units.
    for family, key in [
        ("text", "character_error_rate"),
        ("table", "strict_cell_text_match"),
        ("visual", "region_recall_at_iou_0_5"),
    ]:
        rates = [r[family].get(key) for r in records if r.get(family, {}).get(key) is not None]
        out[family]["page_macro_" + key] = statistics.mean(rates) if rates else None
    return out


def evaluate(reference, packages, output):
    if output.exists():
        raise ValueError("Use a fresh evaluation output directory")
    dataset = json.loads(reference.read_text())
    output.mkdir(parents=True)
    assert len(dataset["pages"]) == 70
    assert all(p["annotation_status"] == "source_transcribed_draft" for p in dataset["pages"])
    for source in dataset["pages"]:
        if digest(reference.parent / source["image"]) != source["image_sha256"]:
            raise ValueError("Reference source image hash mismatch")
    save_atomic(output / "frozen-reference.json", dataset)
    (output / "frozen-scorer.py").write_bytes(Path(__file__).read_bytes())
    refs = defaultdict(list)
    for a in dataset["annotations"]:
        validate_value(a["kind"], a["value"])
        refs[(a["case"], a["page"])].append(a)
    # Explicit uncertainty/deletions stay outside scored reference text.
    exclusions = []
    records = []
    traces = []
    for path in sorted(packages.glob("*/package.json")):
        p = json.loads(path.read_text())
        key = (p["case"], p["page"])
        if key not in refs:
            continue
        page_ref = next(r for r in dataset["pages"] if (r["case"], r["page"]) == key)
        if p["document_id"] != page_ref["document_id"]:
            raise ValueError("Source document mismatch")
        native = Path(p["native_path"]) if p["native_path"] else None
        if native and digest(native) != p["native_sha256"]:
            raise ValueError("Native output changed")
        blocks = p["blocks"]
        items = refs[key]
        row = dict(
            case=key[0],
            page=key[1],
            profile=p["profile"],
            native_present=p["native_present"],
            native_blocks=len(blocks),
            evidence_errors=p["evidence_errors"],
            native_sha256=p["native_sha256"],
        )
        # Geometry-only alignment is chosen before reading any expected text.
        textrefs = []
        for a in items:
            if a["kind"] != "text":
                continue
            uncertain = bool(
                re.search(
                    r"uncertain|unclear|illegible|overwritten|verify|faint|unfinished|"
                    r"excluded|crossed.out|double overbar|bar over",
                    a.get("note", ""),
                    re.I,
                )
            )
            if (
                a["value"].get("role") in {"deleted_text", "diagram_labels", "chart_labels"}
                or uncertain
            ):
                exclusions.append(
                    dict(
                        case=key[0],
                        page=key[1],
                        annotation=a["id"],
                        reason="diagram/chart text scored separately"
                        if a["value"].get("role") in {"diagram_labels", "chart_labels"}
                        else "unresolved or deleted source content",
                    )
                )
                continue
            textrefs.append(a)
        textpred = [
            b
            for b in blocks
            if str(b.get("text", "")).strip() and b["category"].lower() not in VISUAL | {"table"}
        ]
        assignments = assign(textrefs, textpred, 0.15)
        # Source paragraphs split differently by models: assign every overlapping prediction block
        # to its most-overlapped source region, then concatenate in native order.
        owners = defaultdict(list)
        for block in textpred:
            weights = [overlap(a["source_box"], block["source_box"])[0] for a in textrefs]
            if weights and max(weights) > 0:
                j = max(range(len(weights)), key=weights.__getitem__)
                if overlap(textrefs[j]["source_box"], block["source_box"])[1] >= 0.35:
                    owners[j].append(block)
        reftext = "\n".join(normalized(a["value"]["text"]) for a in textrefs)
        predtext = "\n".join(normalized(b["text"]) for bs in owners.values() for b in bs)
        # Page-level content metric neutralizes paragraph alignment for token coverage; CER/WER
        # deliberately include reading-order/segmentation sensitivity and are labelled accordingly.
        orderedpred = sorted(
            [b for bs in owners.values() for b in bs], key=lambda b: b["native_order"]
        )
        predtext = "\n".join(normalized(b["text"]) for b in orderedpred)
        if reftext:
            ar, ap = reftext.split(), predtext.split()
            m = count_match(ar, ap)
            row["text"] = dict(
                reference_characters=len(reftext),
                page_order_character_edits=edits(reftext, predtext),
                character_edits=0,
                reference_words=len(ar),
                page_order_word_edits=edits(ar, ap),
                word_edits=0,
                token_reference=m["reference"],
                token_prediction=m["prediction"],
                token_matched=m["matched"],
            )
            region_traces = []
            for a in textrefs:
                candidates = [
                    b for b in textpred if overlap(a["source_box"], b["source_box"])[1] >= 0.35
                ]
                candidate_text = normalized(" ".join(str(b["text"]) for b in candidates))
                expected_text = normalized(a["value"]["text"])
                ce = edits(expected_text, candidate_text, substring=True)
                we = edits(expected_text.split(), candidate_text.split(), substring=True)
                row["text"]["character_edits"] += ce
                row["text"]["word_edits"] += we
                region_traces.append(
                    dict(
                        annotation=a["id"],
                        expected=expected_text,
                        prediction=candidate_text,
                        character_edits=ce,
                        word_edits=we,
                    )
                )
            row["text"]["reference_characters"] = sum(
                len(normalized(a["value"]["text"])) for a in textrefs
            )
            row["text"]["reference_words"] = sum(
                len(normalized(a["value"]["text"]).split()) for a in textrefs
            )
            row["text"]["character_error_rate"] = ratio(
                row["text"]["character_edits"], row["text"]["reference_characters"]
            )
            traces.append(
                dict(
                    **{k: row[k] for k in ["case", "page", "profile"]},
                    category="text",
                    reference=reftext,
                    prediction=predtext,
                    matched_source_regions=len(owners),
                    reference_regions=len(textrefs),
                    region_traces=region_traces,
                )
            )
        headings = [a for a in textrefs if a["value"].get("role") == "heading"]
        hp = [b for b in blocks if any(s in b["category"].lower() for s in ["title", "heading"])]
        row["heading"] = dict(
            reference_headings=len(headings), matched_headings=len(assign(headings, hp, 0.3))
        )
        pairs = [
            (i, j)
            for i in range(len(textrefs))
            for j in range(i + 1, len(textrefs))
            if i in assignments and j in assignments
        ]
        row["order"] = dict(
            eligible_pairs=len(pairs),
            correct_pairs=sum(
                textpred[assignments[i]]["native_order"] < textpred[assignments[j]]["native_order"]
                for i, j in pairs
            ),
        )
        tables = [a for a in items if a["kind"] == "table_cells"]
        all_predicted_rows = []
        for block in blocks:
            if block["category"].lower() != "table":
                continue
            try:
                _, _, parsed_cells = table_cells(str(block["text"]))
                by_row = defaultdict(list)
                for c in parsed_cells:
                    by_row[c.row].append(c)
                all_predicted_rows.extend(by_row.values())
            except ValueError:
                pass
        tablepred = [b for b in blocks if b["category"].lower() == "table"]
        tm = assign(tables, tablepred, 0.20)
        tot = Counter()
        for i, a in enumerate(tables):
            actual = []
            parse_error = None
            if i in tm:
                try:
                    _, _, actual = table_cells(str(tablepred[tm[i]]["text"]))
                except ValueError as e:
                    parse_error = str(e)
            cells = a["value"]["cells"]
            lookup = {(c.row, c.column): c for c in actual}
            stats = Counter(
                reference_cells=len(cells), reference_tables=1, matched_tables=int(i in tm)
            )
            checks = []
            for c in cells:
                found = lookup.get((c["row"], c["column"]))
                passed = found is not None and normalized(c["text"]) == normalized(found.text)
                stats["exact_cells"] += passed
                stats["span_attributes"] += 2
                stats["span_matches"] += sum(
                    found is not None and c[k] == getattr(found, k)
                    for k in ["row_span", "column_span"]
                )
                stats["header_matches"] += (
                    found is not None and c.get("is_header") == found.is_header
                )
                checks.append(
                    dict(
                        row=c["row"],
                        column=c["column"],
                        expected=c["text"],
                        actual=found.text if found else None,
                        passed=passed,
                    )
                )
            # Label-linked body rows complement strict grid scores.
            ref_by_row = defaultdict(list)
            for c in cells:
                ref_by_row[c["row"]].append(c)
            key_column = 2 if key[0] == "nasa" and key[1] == 17 else 0
            row_checks = []
            key_counts = Counter(
                normalized(c["text"])
                for c in cells
                if c["column"] == key_column
                and c["row_span"] == 1
                and c["column_span"] == 1
                and c["row"] >= a["value"].get("header_rows", 0)
            )
            for rn, ref_cells in ref_by_row.items():
                if rn < a["value"].get("header_rows", 0):
                    continue
                key_cell = next(
                    (
                        c
                        for c in ref_cells
                        if c["column"] == key_column
                        and c["row_span"] == 1
                        and c["column_span"] == 1
                        and c["text"].strip()
                    ),
                    None,
                )
                if not key_cell:
                    continue
                if key_counts[normalized(key_cell["text"])] != 1:
                    stats["nonunique_reference_rows_excluded"] += 1
                    continue
                expected_tail = [
                    normalized(c["text"])
                    for c in sorted(ref_cells, key=lambda c: c["column"])
                    if c["column"] > key_column
                ]
                if not expected_tail:
                    continue
                matches = [
                    tail
                    for predrow in all_predicted_rows
                    for tail in keyed_tail(predrow, key_cell["text"])
                ]
                actual_tail = matches[0][: len(expected_tail)] if len(matches) == 1 else None
                passed = actual_tail == expected_tail
                stats["reference_keyed_rows"] += 1
                stats["keyed_row_matches"] += passed
                row_checks.append(
                    dict(
                        key=key_cell["text"],
                        expected=expected_tail,
                        actual=actual_tail,
                        unique_matches=len(matches),
                        passed=passed,
                    )
                )
            cm = count_match(
                [normalized(c["text"]) for c in cells if c["text"].strip()],
                [normalized(c.text) for c in actual if c.text.strip()],
            )
            stats.update(
                {
                    "content_reference": cm["reference"],
                    "content_prediction": cm["prediction"],
                    "content_matched": cm["matched"],
                }
            )
            tot.update(stats)
            traces.append(
                dict(
                    case=key[0],
                    page=key[1],
                    profile=p["profile"],
                    category="table",
                    annotation=a["id"],
                    prediction_block=tablepred[tm[i]]["id"] if i in tm else None,
                    parse_error=parse_error,
                    checks=checks,
                    row_association_checks=row_checks,
                )
            )
        if tables:
            row["table"] = dict(tot)
            row["table"]["strict_cell_text_match"] = ratio(
                tot["exact_cells"], tot["reference_cells"]
            )
        visuals = [a for a in items if a["kind"] == "layout_region"]
        vp = [b for b in blocks if b["category"].lower() in VISUAL]
        vm = assign(visuals, vp, 0.5)
        if visuals:
            row["visual"] = dict(
                reference_regions=len(visuals),
                matched_regions=len(vm),
                iou_sum=sum(
                    overlap(visuals[i]["source_box"], vp[j]["source_box"])[0] for i, j in vm.items()
                ),
            )
            row["visual"]["region_recall_at_iou_0_5"] = ratio(len(vm), len(visuals))
        values = [a for a in items if a["kind"] == "chart_values"]
        if values:
            total, matched = 0, 0
            for a in values:
                nearby = " ".join(
                    str(b["text"])
                    for b in blocks
                    if overlap(a["source_box"], b["source_box"])[1] >= 0.5
                )
                numbers = re.findall(
                    r"(?<![\w.])[-+]?\d+(?:[ ,]\d{3})*(?:\.\d+)?", normalized(nearby)
                )

                def normalize_number(x):
                    return float(x.replace(",", "").replace(" ", ""))

                actual = Counter(normalize_number(x) for x in numbers)
                expected = Counter(float(pt["value"]) for pt in a["value"]["printed_points"])
                total += sum(expected.values())
                matched += sum((expected & actual).values())
            row["chart"] = dict(reference_values=total, matched_values=matched)
        records.append(row)
    assert len(records) == 280, (len(records), "Expected all four profiles on all 70 pages")
    summary = {
        profile: summarize([r for r in records if r["profile"] == profile])
        for profile in ["paddle", "dots", "bf16", "unlimited"]
    }
    per_pdf = {
        case: {
            profile: summarize(
                [r for r in records if r["case"] == case and r["profile"] == profile]
            )
            for profile in summary
        }
        for case in ["nasa", "math", "irena", "imf"]
    }
    result = dict(
        version="manual-reference-ocr-development-evaluation-v1",
        reference_sha256=digest(reference),
        scorer_sha256=digest(Path(__file__)),
        reference_provenance=dataset.get("provenance"),
        reference_pages=70,
        page_slots=280,
        summary=summary,
        per_pdf=per_pdf,
        limitations=[
            (
                "Assistant-created retrospective source references, not independent human "
                "ground truth. User endorsed the approach and inspected examples; no "
                "per-annotation human approvals are implied."
            ),
            (
                "Text CER/WER use best contiguous matches inside geometrically overlapping "
                "native blocks. They measure scoped content recovery, not hallucination "
                "precision. Page-order edit diagnostics are retained separately. Token "
                "F1 is order-insensitive and case-sensitive. Conservative formatting "
                "normalization preserves signs and numbers."
            ),
            (
                "Strict table-grid origin/span metrics penalize alternative table layouts. "
                "Label-linked row checks validate uniquely matched keys and ordered values, "
                "tolerating header offsets. Cell-text multiset F1 is complementary and does "
                "not validate country/value "
                "associations."
            ),
            (
                "Heading metric measures region detection, not heading levels or parent "
                "correctness. Pair-order metric covers only matched text-region pairs; "
                "denominator is reported."
            ),
            (
                "Visual metric measures region recall and crop overlap only. "
                "Logos/decorations are outside reference scope; no false-positive precision "
                "or semantic image-description score is claimed."
            ),
            (
                "Chart metric is numeric token recall in the source chart region, not "
                "series/category association accuracy. Axis numbers can produce matches; this "
                "diagnostic must not be called chart understanding."
            ),
            (
                "Raw native predictions only; Gemma descriptions, native PDF image recovery "
                "and pipeline repairs are excluded."
            ),
            (
                "Unresolved handwriting/deleted passages are excluded. Text normalization "
                "does not attempt complete LaTeX equivalence; formula semantic accuracy is "
                "not measured."
            ),
            (
                "Dots BF16 used Colab T4 while local profiles used GTX 1660 Ti 6 GiB. Scores "
                "do not isolate quantization or establish a universal winner."
            ),
        ],
    )
    save_atomic(output / "metrics.json", result)
    save_atomic(output / "page-metrics.json", records)
    save_atomic(output / "failure-traces.json", traces)
    save_atomic(
        output / "excluded-references.json",
        list({json.dumps(x, sort_keys=True): x for x in exclusions}.values()),
    )
    print(
        json.dumps(
            {
                profile: {
                    "CER": v["text"].get("character_error_rate"),
                    "cell_match": v["table"].get("strict_cell_text_match"),
                    "visual_recall": v["visual"].get("region_recall_at_iou_0_5"),
                }
                for profile, v in summary.items()
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--packages",
        type=Path,
        default=ROOT / "outputs/quality_evaluations/raw-ocr-vlm-judge-01/packages",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.reference, args.packages, args.output_directory)
