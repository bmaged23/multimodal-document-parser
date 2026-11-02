"""Correlate printed line-chart endpoints with native PDF paths and legend colors.

This deliberately narrow extractor abstains for raster charts, transformed forms,
nonlinear axes, missing legends, ambiguous labels, and non-year categories. It
never uses model point values, evaluation references, or values inferred from
heights: every returned value is a native printed text object.
"""

import ctypes
import re
from difflib import SequenceMatcher
from math import dist
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium  # type: ignore[import-untyped]

from multimodal_doc_parser.document_schema.document import ChartElement, ChartSeries


def color(obj: Any, fill: bool) -> tuple[int, ...]:
    channels = [ctypes.c_uint() for _ in range(4)]
    getter = pdfium.raw.FPDFPageObj_GetFillColor if fill else pdfium.raw.FPDFPageObj_GetStrokeColor
    getter(obj.raw, *[ctypes.byref(value) for value in channels])
    return tuple(c.value for c in channels)


def points(obj: Any) -> list[tuple[float, float]]:
    count = pdfium.raw.FPDFPath_CountSegments(obj.raw)
    output = []
    for index in range(count):
        segment = pdfium.raw.FPDFPath_GetPathSegment(obj.raw, index)
        if pdfium.raw.FPDFPathSegment_GetType(segment) not in {
            pdfium.raw.FPDF_SEGMENT_MOVETO,
            pdfium.raw.FPDF_SEGMENT_LINETO,
        } or pdfium.raw.FPDFPathSegment_GetClose(segment):
            return []
        x, y = ctypes.c_float(), ctypes.c_float()
        if not pdfium.raw.FPDFPathSegment_GetPoint(segment, ctypes.byref(x), ctypes.byref(y)):
            return []
        output.append(obj.get_matrix().on_point(x.value, y.value))
    return output


def recover_vector_endpoints(element: ChartElement, bundle: Path) -> dict[str, Any] | None:
    analysis = element.description.analysis if element.description else None
    if not analysis or not 3 <= len(analysis.series) <= 12:
        return None
    if analysis.plot_type not in {None, "line"}:
        return None
    region = element.source_regions[0]
    with pdfium.PdfDocument(bundle / "source.pdf") as pdf:
        page = pdf[region.page_number - 1]
        try:
            if page.get_rotation() != 0:
                return None
            width, height = page.get_size()
            b = region.bounding_box
            bounds = (b.x0 * width, (1 - b.y1) * height, b.x1 * width, (1 - b.y0) * height)
            textpage = page.get_textpage()
            try:
                objects = list(page.get_objects(textpage=textpage))
                texts = []
                paths = []
                for obj in objects:
                    box = obj.get_bounds()
                    if not (
                        bounds[0] <= box[0] <= box[2] <= bounds[2]
                        and bounds[1] <= box[1] <= box[3] <= bounds[3]
                    ):
                        continue
                    if obj.type == pdfium.raw.FPDF_PAGEOBJ_TEXT:
                        text = obj.extract().strip()
                        if text:
                            texts.append(
                                dict(
                                    text=text,
                                    box=box,
                                    center=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                                )
                            )
                    elif obj.type == pdfium.raw.FPDF_PAGEOBJ_PATH and obj.level == 0:
                        paths.append(
                            dict(obj=obj, box=box, fill=color(obj, True), stroke=color(obj, False))
                        )
                return correlate(analysis.series, texts, paths, bounds)
            finally:
                textpage.close()
        finally:
            page.close()


def correlate(
    series: list[ChartSeries],
    texts: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    bounds: tuple[float, ...],
) -> dict[str, Any] | None:
    curves = []
    for path in paths:
        box = path["box"]
        if box[2] - box[0] < (bounds[2] - bounds[0]) * 0.5 or box[3] - box[1] < 4:
            continue
        vertices = points(path["obj"])
        if len(vertices) >= 3 and vertices[-1][0] > vertices[0][0]:
            curves.append(dict(**path, first=vertices[0], last=vertices[-1]))
    if len(curves) != len(series):
        return None
    xmin = sum(c["first"][0] for c in curves) / len(curves)
    xmax = sum(c["last"][0] for c in curves) / len(curves)
    if any(abs(c["first"][0] - xmin) > 1.5 or abs(c["last"][0] - xmax) > 1.5 for c in curves):
        return None
    # Legend swatches must have adjacent nonnumeric text on their right.
    legends = []
    for path in paths:
        x0, y0, x1, y1 = path["box"]
        if not 4 <= x1 - x0 <= 14 or not 4 <= y1 - y0 <= 14:
            continue
        labels = [
            t
            for t in texts
            if 0 <= t["box"][0] - x1 <= 12
            and abs(t["center"][1] - (y0 + y1) / 2) < 4
            and re.search("[A-Za-z]", t["text"])
            and len(t["text"]) < 60
        ]
        if len(labels) == 1:
            item = dict(name=labels[0]["text"], color=path["fill"][:3])
            if item not in legends:
                legends.append(item)
    if len(legends) != len(series):
        return None

    # Exact labels first; then allow one uniquely similar text-encoding/ligature
    # fallback only after exact matches have removed competing series names.
    def key(value: str) -> str:
        return re.sub("[^a-z0-9]", "", value.casefold())

    matches: dict[str, dict[str, Any]] = {}
    remaining = list(legends)
    for candidate_series in series:
        exact = [v for v in remaining if key(v["name"]) == key(candidate_series.name)]
        if len(exact) == 1:
            matches[candidate_series.name] = exact[0]
            remaining.remove(exact[0])
    unmatched = [s for s in series if s.name not in matches]
    if unmatched:
        if len(unmatched) != 1 or len(remaining) != 1:
            return None
        if SequenceMatcher(None, key(unmatched[0].name), key(remaining[0]["name"])).ratio() < 0.75:
            return None
        matches[unmatched[0].name] = remaining[0]
    for curve in curves:
        nearest = sorted(
            (dist(curve["stroke"][:3], legend["color"]), name) for name, legend in matches.items()
        )
        if nearest[0][0] > 90 or nearest[1][0] - nearest[0][0] < 15:
            return None
        curve["series"] = nearest[0][1]
    if len({c["series"] for c in curves}) != len(curves):
        return None
    numeric = []
    for text in texts:
        if "," in text["text"]:
            continue  # Locale-dependent decimal/thousands notation is ambiguous.
        cleaned = text["text"].replace("−", "-").replace(" ", "").replace(",", "")
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
            numeric.append(dict(**text, value=float(cleaned)))
    horizontal = [
        p
        for p in paths
        if p["box"][2] - p["box"][0] >= xmax - xmin and p["box"][3] - p["box"][1] <= 3
    ]
    ticks = []
    for text in numeric:
        if text["box"][2] >= xmin - 10:
            continue
        planes = [
            (p["box"][1] + p["box"][3]) / 2
            for p in horizontal
            if abs((p["box"][1] + p["box"][3]) / 2 - text["center"][1]) < 3
        ]
        if planes:
            ticks.append((text["value"], min(planes, key=lambda y: abs(y - text["center"][1]))))
    if len(ticks) < 3:
        return None
    avg_x = sum(t[0] for t in ticks) / len(ticks)
    avg_y = sum(t[1] for t in ticks) / len(ticks)
    denominator = sum((t[0] - avg_x) ** 2 for t in ticks)
    if denominator == 0:
        return None
    slope = sum((x - avg_x) * (y - avg_y) for x, y in ticks) / denominator
    intercept = avg_y - slope * avg_x
    if slope <= 0 or max(abs(y - (slope * x + intercept)) for x, y in ticks) > 1:
        return None
    recovered = []
    for endpoint, x in [("first", xmin), ("last", xmax)]:
        years = [
            t
            for t in numeric
            if re.fullmatch(r"\d{4}", t["text"])
            and abs(t["center"][0] - x) < 5
            and intercept - 20 < t["center"][1] < intercept
        ]
        labels = [
            t for t in numeric if abs(t["center"][0] - x) < 30 and t["center"][1] > intercept + 3
        ]
        if len(years) != 1 or len(labels) != len(curves):
            return None
        ordered_curves = sorted(curves, key=lambda c: c[endpoint][1])
        ordered_labels = sorted(labels, key=lambda t: t["value"])
        for curve, label in zip(ordered_curves, ordered_labels, strict=True):
            projected = (curve[endpoint][1] - intercept) / slope
            if abs(projected - label["value"]) > 2 / slope:
                return None
            recovered.append(
                dict(
                    series=curve["series"],
                    label=years[0]["text"],
                    value=label["value"],
                    printed_text=label["text"],
                    printed_text_box=label["box"],
                    path_endpoint=curve[endpoint],
                    legend=matches[curve["series"]],
                    value_basis="printed_label",
                )
            )
    categories = {p["label"] for p in recovered}
    if any(p.value is not None and p.label not in categories for s in series for p in s.points):
        return None
    return dict(
        method="native_pdf_line_path_legend_and_printed_endpoint_correlation",
        point_values="Copied only from native printed text objects; geometry is used for "
        "association and a consistency check, never to generate values.",
        axis_calibration=dict(slope=slope, intercept=intercept),
        points=recovered,
    )
