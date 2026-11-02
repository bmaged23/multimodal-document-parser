"""Render canonical content to safe HTML fragments and readable Markdown."""

import re
from html import escape

from multimodal_doc_parser.document_schema.document import (
    ChartElement,
    Document,
    DocumentElement,
    FormulaElement,
    HeadingElement,
    ImageElement,
    TableElement,
    TextElement,
)


def inline_source_html(text: str) -> str:
    """Render only recognized OCR footnote superscripts, never arbitrary model HTML/math."""
    escaped = escape(text)
    return re.sub(
        r"\$\s*([A-Za-z ]*?)\s*\^\s*\{{1,2}(\d{1,3})\}{1,2}\s*\$",
        lambda match: match[1].strip() + f"<sup>{match[2]}</sup>",
        escaped,
    )


def visual_analysis_html(element: ChartElement | ImageElement) -> str:
    description = element.description
    if not description:
        return ""
    provenance = description.provenance
    parts = [
        f"<p><strong>{escape(provenance.method.title())} visual interpretation</strong> "
        f"— {escape(provenance.producer)} {escape(provenance.model_version or '')}</p>"
    ]
    if element.quality_review:
        review = element.quality_review
        parts.append(
            f"<p><strong>Visual quality: {escape(review.status.replace('_', ' '))}.</strong> "
            "Model review is not independent source verification.</p>"
        )
        if review.warnings:
            parts.append(
                "<ul>" + "".join(f"<li>{escape(w)}</li>" for w in review.warnings) + "</ul>"
            )
        if review.adjustments:
            parts.append(
                "<details><summary>Source checks and adjustments</summary><ul>"
                + "".join(f"<li>{escape(w)}</li>" for w in review.adjustments)
                + "</ul></details>"
            )
    analysis = description.analysis
    if analysis:
        for label, value in [
            ("Title", analysis.title),
            ("X axis", analysis.x_axis),
            ("Y axis", analysis.y_axis),
        ]:
            if value:
                parts.append(f"<p>{label}: {escape(value)}</p>")
        extracted_series = (
            isinstance(element, ChartElement)
            and element.series
            and (
                element.series_provenance is None or element.series_provenance.method != "generated"
            )
        )
        numeric_series = (
            element.series
            if isinstance(element, ChartElement) and extracted_series
            else analysis.series
        )
        if numeric_series:
            rows = []
            for series in numeric_series:
                for point in series.points:
                    value = str(point.value) if point.value is not None else "Unknown"
                    rows.append(
                        "<tr>"
                        + "".join(
                            f"<td>{escape(item)}</td>"
                            for item in [
                                series.name,
                                point.label,
                                value,
                                series.unit or "",
                                point.value_basis,
                            ]
                        )
                        + "</tr>"
                    )
            source_values = (
                isinstance(element, ChartElement)
                and element.series_provenance
                and (element.series_provenance.method == "extracted")
            )
            if source_values and isinstance(element, ChartElement) and element.series_provenance:
                value_label = (
                    "Source-extracted chart values — "
                    + escape(element.series_provenance.producer)
                    + "."
                )
            elif extracted_series:
                value_label = (
                    "Existing chart values — retain their recorded provenance; verify the source."
                )
            else:
                value_label = "Model-reported chart values — verify against the source."
            parts.append(
                f"<p>{value_label}</p>"
                "<table><tr><th>Series</th><th>Category</th><th>Value</th>"
                "<th>Unit</th><th>Basis</th></tr>" + "".join(rows) + "</table>"
            )
        if analysis.nodes:
            nodes = {node.id: node.label for node in analysis.nodes}
            parts.append(
                "<p>Flowchart nodes</p><ul>"
                + "".join(f"<li>{escape(n.label)}</li>" for n in analysis.nodes)
                + "</ul>"
            )
            parts.append(
                "<p>Connections</p><ul>"
                + "".join(
                    f"<li>{escape(nodes[e.source])} → {escape(nodes[e.target])}"
                    f"{(' (' + escape(e.label) + ')') if e.label else ''}</li>"
                    for e in analysis.edges
                )
                + "</ul>"
            )
        if analysis.uncertainties:
            parts.append(
                "<p>Uncertainty</p><ul>"
                + "".join(f"<li>{escape(item)}</li>" for item in analysis.uncertainties)
                + "</ul>"
            )
    return "".join(parts)


def element_html(element: DocumentElement) -> str:
    if isinstance(element, HeadingElement):
        return f"<h{element.level}>{escape(element.text)}</h{element.level}>"
    if isinstance(element, TextElement):
        return f'<p class="preserve-lines">{inline_source_html(element.text)}</p>'
    if isinstance(element, FormulaElement):
        return f'<pre class="formula">{escape(element.latex)}</pre>'
    if isinstance(element, TableElement):
        cells = {(cell.row, cell.column): cell for cell in element.cells}
        covered = {
            (row, column)
            for cell in element.cells
            for row in range(cell.row, cell.row + cell.row_span)
            for column in range(cell.column, cell.column + cell.column_span)
            if (row, column) != (cell.row, cell.column)
        }
        rows = []
        for row in range(element.row_count):
            parts = []
            for column in range(element.column_count):
                if (row, column) in covered:
                    continue
                cell = cells.get((row, column))
                if cell is None:
                    parts.append('<td class="missing" title="Cell not extracted">—</td>')
                    continue
                tag = "th" if cell.is_header else "td"
                parts.append(
                    f'<{tag} rowspan="{cell.row_span}" colspan="{cell.column_span}">'
                    f"{inline_source_html(cell.text)}</{tag}>"
                )
            rows.append("<tr>" + "".join(parts) + "</tr>")
        caption = f"<caption>{escape(element.caption)}</caption>" if element.caption else ""
        note = "<p>Header roles inferred; check the source.</p>" if element.header_inference else ""
        if element.continuation_index and element.continuation_index > 1:
            note += (
                f"<p>Table fragment {element.continuation_index}"
                f" — {escape(element.column_group or 'continued')}.</p>"
            )
        return (
            note + '<div class="table-scroll"><table>' + caption + "".join(rows) + "</table></div>"
        )
    if isinstance(element, (ChartElement, ImageElement)):
        caption = escape(element.caption or element.type.title())
        description = (
            escape(element.description.text)
            if element.description
            else ("Visual description pending enrichment.")
        )
        return (
            f'<figure><img loading="lazy" src="{escape(element.asset.path, quote=True)}" '
            f'alt="{caption}"><figcaption>{caption}</figcaption></figure><p>{description}</p>'
            + visual_analysis_html(element)
        )
    raise TypeError(f"Unsupported element type: {type(element)}")


def document_markdown(document: Document, *, page_number: int | None = None) -> str:
    title = document.title or document.source_filename
    parts = [f"# {escape(title)}", ""] if page_number is None else []
    for element in document.elements:
        if page_number is not None and not any(
            region.page_number == page_number for region in element.source_regions
        ):
            continue
        if isinstance(element, HeadingElement):
            content = f"{'#' * element.level} {escape(element.text)}"
        elif isinstance(element, TextElement):
            content = inline_source_html(element.text)
        elif isinstance(element, FormulaElement):
            content = f"$$\n{element.latex}\n$$"
        elif isinstance(element, TableElement):
            # HTML preserves merged cells, which pipe tables cannot express reliably.
            content = element_html(element)
        elif isinstance(element, (ChartElement, ImageElement)):
            content = f"![{element.type}]({element.asset.path})"
            if element.caption:
                content += "\n\n" + escape(element.caption)
            if element.description:
                content += "\n\n" + escape(element.description.text)
                content += "\n\n" + visual_analysis_html(element)
        else:
            raise TypeError(f"Unsupported element type: {type(element)}")
        pages = ", ".join(str(region.page_number) for region in element.source_regions)
        parts.extend([content, "", f"<!-- source: {element.id}; pages: {pages} -->", ""])
    return "\n".join(parts)
