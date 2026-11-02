"""Native extraction must use printed labels, and abstain when evidence is incomplete."""

from pathlib import Path

from reportlab.lib.colors import Color
from reportlab.pdfgen.canvas import Canvas

from multimodal_doc_parser.document_schema.document import ChartElement
from multimodal_doc_parser.visual_enrichment.vector_endpoints import recover_vector_endpoints


def synthetic_chart(tmp_path: Path, omit_label: bool = False) -> ChartElement:
    canvas = Canvas(str(tmp_path / "source.pdf"), pagesize=(300, 300))
    canvas.setFont("Helvetica", 8)
    for value in [0, 50, 100]:
        y = 60 + 2 * value
        canvas.setStrokeColorRGB(0.7, 0.7, 0.7)
        canvas.line(35, y, 270, y)
        canvas.setFillColorRGB(0, 0, 0)
        canvas.drawRightString(25, y - 3, str(value))
    canvas.drawCentredString(60, 44, "2010")
    canvas.drawCentredString(240, 44, "2024")
    cases = [
        ("Red", (1, 0, 0), 20, 30),
        ("Green", (0, 0.7, 0), 40, 50),
        ("Blue", (0, 0, 1), 60, 80),
    ]
    for index, (name, rgb, first, last) in enumerate(cases):
        canvas.setStrokeColor(Color(*rgb))
        path = canvas.beginPath()
        path.moveTo(60, 60 + 2 * first)
        path.lineTo(150, 60 + first + last)
        path.lineTo(240, 60 + 2 * last)
        canvas.drawPath(path)
        canvas.setFillColorRGB(0, 0, 0)
        canvas.drawCentredString(60, 65 + 2 * first, str(first))
        if not (omit_label and index == 2):
            canvas.drawCentredString(240, 65 + 2 * last, str(last))
        x = 60 + index * 70
        canvas.setFillColor(Color(*rgb))
        canvas.circle(x, 30, 3, stroke=0, fill=1)
        canvas.setFillColorRGB(0, 0, 0)
        canvas.drawString(x + 9, 27, name)
    canvas.save()
    return ChartElement(
        id="synthetic",
        source_regions=[dict(page_number=1, bounding_box=dict(x0=0.02, y0=0.02, x1=0.98, y1=0.98))],
        provenance=dict(producer="fixture", method="extracted"),
        asset=dict(path="unused.png"),
        description=dict(
            text="A synthetic line chart",
            provenance=dict(producer="fixture", method="generated"),
            analysis=dict(
                kind="chart",
                plot_type="line",
                description="A line chart",
                series=[
                    dict(
                        name=name,
                        unit="units",
                        points=[dict(label="2024", value=999, value_basis="printed_label")],
                    )
                    for name, _, _, _ in cases
                ],
            ),
        ),
    )


def test_correlates_labels_with_color_paths_without_using_model_values(tmp_path):
    element = synthetic_chart(tmp_path)
    result = recover_vector_endpoints(element, tmp_path)
    assert result is not None
    values = {(p["series"], p["label"]): p["value"] for p in result["points"]}
    assert values == {
        ("Red", "2010"): 20,
        ("Red", "2024"): 30,
        ("Green", "2010"): 40,
        ("Green", "2024"): 50,
        ("Blue", "2010"): 60,
        ("Blue", "2024"): 80,
    }
    assert all(p["printed_text_box"] and p["legend"] for p in result["points"])
    assert element.description.analysis.series[0].points[0].value == 999


def test_missing_printed_label_never_becomes_an_estimated_value(tmp_path):
    element = synthetic_chart(tmp_path, omit_label=True)
    assert recover_vector_endpoints(element, tmp_path) is None


def test_ambiguous_legend_does_not_trigger_source_correction(tmp_path):
    element = synthetic_chart(tmp_path)
    element.description.analysis.series[0].name = "Unknown A"
    element.description.analysis.series[1].name = "Unknown B"
    assert recover_vector_endpoints(element, tmp_path) is None


def source_document(element):
    from multimodal_doc_parser.document_schema.document import Document

    return Document(
        document_id="sha256:" + "0" * 64,
        source_filename="source.pdf",
        status="parsed",
        preparation=dict(dpi=72, renderer="fixture"),
        pages=[
            dict(
                page_number=1,
                width_points=300,
                height_points=300,
                width_pixels=300,
                height_pixels=300,
                rotation_degrees=0,
                image=dict(path="unused.png"),
            )
        ],
        elements=[element],
    )


def test_source_summary_and_numeric_provenance_are_distinct_from_model_output(tmp_path):
    from multimodal_doc_parser.visual_enrichment.source_checks import finalize_visual_source_checks

    element = synthetic_chart(tmp_path)
    element.description.analysis.description = "All series decline."
    document = source_document(element)
    finalize_visual_source_checks(document, element, tmp_path, tmp_path / "raw", "fixture-model")
    assert element.quality_review.status == "source_correlated"
    assert element.series_provenance.method == "extracted"
    assert "increase" in element.description.text and "decline" not in element.description.text
    assert element.description.provenance.producer == "PDF source endpoint summary"
    assert (tmp_path / "raw/synthetic-native-endpoint-evidence.json").exists()


def test_existing_extracted_chart_values_are_preserved(tmp_path):
    from multimodal_doc_parser.document_exports.readable_exports import visual_analysis_html
    from multimodal_doc_parser.document_schema.document import ChartSeries, Provenance
    from multimodal_doc_parser.visual_enrichment.source_checks import finalize_visual_source_checks

    element = synthetic_chart(tmp_path)
    element.series = [
        ChartSeries(
            name="Red",
            unit="units",
            points=[dict(label="2010", value=20, value_basis="printed_label")],
        )
    ]
    element.series_provenance = Provenance(producer="fixture source", method="extracted")
    finalize_visual_source_checks(
        source_document(element), element, tmp_path, tmp_path / "raw", "model"
    )
    assert element.series[0].points[0].value == 20
    assert element.series_provenance.producer == "fixture source"
    assert ">999.0<" not in visual_analysis_html(element)
