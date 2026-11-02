"""Source-only chart evidence and transparent post-generation quality checks."""

import json
import re
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium  # type: ignore[import-untyped]

from multimodal_doc_parser.document_schema.document import (
    ChartElement,
    ChartPoint,
    Document,
    ImageElement,
    Provenance,
    VisualAnalysis,
    VisualQualityReview,
)


def source_text(
    document: Document, element: ChartElement | ImageElement, bundle: Path
) -> dict[str, Any]:
    """Read native text in the target and directly below it, never evaluation references."""
    region = element.source_regions[0]
    with pdfium.PdfDocument(bundle / "source.pdf") as pdf:
        page = pdf[region.page_number - 1]
        try:
            if page.get_rotation() != 0:
                return {"target": "", "below_target": ""}
            width, height = page.get_size()
            b = region.bounding_box
            text = page.get_textpage()
            try:
                target = text.get_text_bounded(
                    b.x0 * width, (1 - b.y1) * height, b.x1 * width, (1 - b.y0) * height
                )
                below = text.get_text_bounded(
                    b.x0 * width,
                    max(0, 1 - b.y1 - 0.07) * height,
                    b.x1 * width,
                    (1 - b.y1) * height,
                )
                numeric_objects = []
                positioned_numbers = []
                for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_TEXT], textpage=text):
                    x0, y0, x1, y1 = obj.get_bounds()
                    if (
                        b.x0 * width <= x0 <= x1 <= b.x1 * width
                        and (1 - b.y1) * height <= y0 <= y1 <= (1 - b.y0) * height
                    ):
                        token = obj.extract().strip()
                        compact = re.sub(r"\s+", "", token).replace("−", "-").rstrip("%")
                        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", compact):
                            numeric_objects.append(token)
                            positioned_numbers.append(dict(text=token, box=(x0, y0, x1, y1)))
                joined = join_numeric_groups(positioned_numbers)
                numeric_objects.extend(joined)
            finally:
                text.close()
        finally:
            page.close()
    return {"target": target, "below_target": below, "numeric_text_objects": numeric_objects}


def join_numeric_groups(items: list[dict[str, Any]]) -> list[str]:
    """Join separate thousands-group glyph objects only when spatially adjacent."""
    output = []
    for first in items:
        if not re.fullmatch(r"[-+]?\d{1,3}", first["text"]):
            continue
        current = first
        parts = [first["text"]]
        for _ in range(3):
            x0, y0, x1, y1 = current["box"]
            next_items = [
                item
                for item in items
                if re.fullmatch(r"\d{3}", item["text"])
                and 0 <= item["box"][0] - x1 < 3
                and abs((item["box"][1] + item["box"][3]) / 2 - (y0 + y1) / 2) < 1
            ]
            if len(next_items) != 1:
                break
            current = next_items[0]
            parts.append(current["text"])
            output.append(" ".join(parts))
    return output


def apply_source_checks(
    analysis: VisualAnalysis, evidence: dict[str, Any], reviewer: str | None = None
) -> VisualQualityReview:
    review = VisualQualityReview(
        status="model_reviewed" if reviewer else "unverified", reviewer=reviewer
    )
    if analysis.kind != "chart":
        return review
    scatter = analysis.plot_type == "scatter" or bool(
        re.match(r"^(?:a\s+)?scatter\s+plot\b", analysis.description, re.I)
    )
    if scatter and all(p.value is None for s in analysis.series for p in s.points):
        removed = sum(len(s.points) for s in analysis.series)
        for series in analysis.series:
            series.points = []
        if removed:
            review.adjustments.append(
                f"Removed {removed} null-valued scatter categories: "
                "axis ticks are not observed points."
            )
    # Only use an explicit nearby source definition, not the surrounding prose's
    # numeric values or a unit chosen by the model reviewer.
    definitions = re.findall(
        r"\b(GW|MW|kW|ms|bps)\s*=\s*([A-Za-z ]+)", evidence.get("below_target", "")
    )
    units = {unit for unit, _ in definitions}
    if len(units) == 1:
        unit = units.pop()
        spelling = next(
            definition.strip().casefold() for symbol, definition in definitions if symbol == unit
        )
        for series in analysis.series:
            if series.unit is None or series.unit.strip().casefold().rstrip(".") == spelling:
                previous_unit = series.unit
                series.unit = unit
                review.adjustments.append(
                    f"Series {series.name!r}: normalized unit {previous_unit!r} to {unit} "
                    "from the explicit native-PDF note."
                )
    if any(s.points and not s.unit for s in analysis.series):
        review.warnings.append("Chart has data points without a legible unit; verify the source.")
    keys = [
        (s.name.casefold().strip(), p.label.casefold().strip())
        for s in analysis.series
        for p in s.points
    ]
    if len(keys) != len(set(keys)):
        review.warnings.append("Duplicate series/category keys; chart records need source review.")
    # A PDF text layer can corroborate occurrence of a number, not its assignment
    # to a series. Missing text-layer support remains a warning, never auto-repair.
    native = evidence.get("target", "").replace("−", "-")
    if "numeric_text_objects" in evidence:
        # Separate text objects keep adjacent labels such as "691 599 534"
        # distinct instead of mistaking them for a thousands-grouped integer.
        numbers = {
            float(re.sub(r"\s+", "", token).replace("−", "-").rstrip("%"))
            for token in evidence["numeric_text_objects"]
        }
    elif native.strip():
        numbers = {
            float(value.replace(" ", "").replace("\u00a0", "").replace(",", ""))
            for value in re.findall(
                r"(?<![\w.])[-+]?(?:\d{1,3}(?:[ ,\u00a0]\d{3})+|\d+)(?:\.\d+)?(?![\w.])", native
            )
            if re.fullmatch(r"[-+]?\d[\d ,\u00a0]*(?:\.\d+)?", value)
        }
    else:
        numbers = set()
    if native.strip():
        unsupported = [
            (s.name, p.label, p.value)
            for s in analysis.series
            for p in s.points
            if p.value is not None and p.value not in numbers
        ]
        if unsupported:
            review.warnings.append(
                f"{len(unsupported)} numeric point(s) lack matching native-PDF text tokens; "
                "check source labels. Text-token matches do not verify series assignment."
            )
    if review.warnings:
        review.status = "needs_review"
    return review


def finalize_visual_source_checks(
    document: Document,
    element: ChartElement | ImageElement,
    bundle: Path,
    raw: Path,
    reviewer: str | None,
) -> None:
    """Correlate native vector evidence where possible; keep raw model output separate."""
    if not element.description or not element.description.analysis:
        return
    from multimodal_doc_parser.visual_enrichment.vector_endpoints import recover_vector_endpoints

    analysis = element.description.analysis
    evidence = source_text(document, element, bundle)
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{element.id}-source-text-evidence-v2.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    preserve_extracted = (
        isinstance(element, ChartElement)
        and bool(element.series)
        and (element.series_provenance is None or element.series_provenance.method != "generated")
    )
    native = (
        recover_vector_endpoints(element, bundle)
        if isinstance(element, ChartElement) and not preserve_extracted
        else None
    )
    if native and isinstance(element, ChartElement):
        (raw / f"{element.id}-native-endpoint-evidence.json").write_text(
            json.dumps(native, indent=2) + "\n"
        )
        for series in analysis.series:
            series.points = [
                ChartPoint(label=p["label"], value=p["value"], value_basis="printed_label")
                for p in native["points"]
                if p["series"] == series.name
            ]
        element.series_provenance = Provenance(
            producer="PDF native vector/text correlation",
            model_version="endpoint-recovery-v1",
            method="extracted",
        )
        # Model prose can contradict correct endpoint values (for example, claim
        # every series declines). Use a disclosed source-based endpoint summary
        # for this narrowly supported chart type; raw model prose remains saved.
        years = sorted({p["label"] for p in native["points"]}, key=int)
        directions: dict[str, list[str]] = {"decrease": [], "increase": [], "remain equal": []}
        for series in analysis.series:
            values = sorted(series.points, key=lambda p: int(p.label))
            start, end = values[0].value, values[-1].value
            if start is not None and end is not None:
                direction = (
                    "decrease" if end < start else "increase" if end > start else "remain equal"
                )
                directions[direction].append(series.name)
        statements = [
            ", ".join(names) + f" {direction}" for direction, names in directions.items() if names
        ]
        analysis.description = (
            f"Line chart with {len(analysis.series)} series from {years[0]} "
            f"to {years[-1]}. Comparing the printed endpoints: " + "; ".join(statements) + "."
        )
        element.description.text = analysis.description
        element.description.provenance = Provenance(
            producer="PDF source endpoint summary",
            model_version="endpoint-summary-v1",
            method="generated",
        )
    prior = element.quality_review
    review = apply_source_checks(analysis, evidence, reviewer)
    if prior:
        review.adjustments = list(dict.fromkeys(prior.adjustments + review.adjustments))
        review.warnings.extend(w for w in prior.warnings if w.startswith("Chart review failed:"))
    if native:
        review.adjustments.append(
            "Endpoint values were recovered from native printed PDF text, "
            "matched to source line paths, year labels and legend colors. "
            "No evaluation reference or model point value was used."
        )
        review.adjustments.append(
            "Description replaced by a source-based endpoint comparison; "
            "raw VLM descriptions remain in their response files."
        )
        if not review.warnings:
            review.status = "source_correlated"
    if review.warnings:
        review.status = "needs_review"
    element.quality_review = review
    if isinstance(element, ChartElement) and not preserve_extracted:
        element.series = analysis.series


def flag_conflicting_chart_values(document: Document) -> None:
    """Flag disagreement across repeated series/category/unit values; never choose a winner."""
    charts = [
        e
        for e in document.elements
        if isinstance(e, ChartElement) and e.description and e.description.analysis
    ]
    for index, left in enumerate(charts):
        left_analysis = left.description.analysis if left.description else None
        if left_analysis is None:
            continue
        left_keys = {
            (s.name.casefold().strip(), p.label.casefold().strip(), s.unit): p.value
            for s in left_analysis.series
            for p in s.points
            if p.value is not None
        }
        for right in charts[index + 1 :]:
            right_analysis = right.description.analysis if right.description else None
            if right_analysis is None:
                continue
            right_keys = {
                (s.name.casefold().strip(), p.label.casefold().strip(), s.unit): p.value
                for s in right_analysis.series
                for p in s.points
                if p.value is not None
            }
            common = left_keys.keys() & right_keys.keys()
            # Require substantial overlap so unrelated charts sharing one year
            # or category aren't treated as duplicate figures.
            if len(common) < 4:
                continue
            conflicts = [key for key in common if left_keys[key] != right_keys[key]]
            if conflicts:
                for current, other in [(left, right), (right, left)]:
                    review = current.quality_review or VisualQualityReview()
                    review.status = "needs_review"
                    review.warnings.append(
                        f"{len(conflicts)} repeated series/category values "
                        f"differ from chart {other.id}; verify both sources."
                    )
                    current.quality_review = review
