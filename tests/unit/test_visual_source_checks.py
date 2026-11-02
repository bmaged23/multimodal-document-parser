"""Source guards must disclose uncertainty rather than invent numerical repairs."""

from multimodal_doc_parser.document_schema.document import VisualAnalysis
from multimodal_doc_parser.visual_enrichment.source_checks import apply_source_checks


def chart(plot="scatter", value=None, unit=None):
    return VisualAnalysis(
        kind="chart",
        plot_type=plot,
        description="A synthetic chart",
        series=[
            dict(
                name="capacity",
                unit=unit,
                points=[
                    dict(
                        label="0.2",
                        value=value,
                        value_basis="unknown" if value is None else "printed_label",
                    )
                ],
            )
        ],
    )


def test_axis_ticks_are_removed_only_for_unlabelled_scatter_plots():
    scatter = chart()
    review = apply_source_checks(scatter, {})
    assert scatter.series[0].points == [] and review.adjustments
    bar = chart("bar")
    apply_source_checks(bar, {})
    assert len(bar.series[0].points) == 1


def test_units_are_restored_only_from_unambiguous_explicit_source_notes():
    analysis = chart("bar", 70)
    review = apply_source_checks(analysis, dict(target="70", below_target="Note: GW = gigawatt."))
    assert analysis.series[0].unit == "GW"
    assert analysis.series[0].points[0].value == 70
    assert "native-PDF" in review.adjustments[0]
    ambiguous = chart("bar", 70)
    review = apply_source_checks(
        ambiguous, dict(target="70", below_target="GW = gigawatt; MW = megawatt")
    )
    assert ambiguous.series[0].unit is None and review.status == "needs_review"


def test_unsupported_numbers_are_flagged_without_modification():
    analysis = chart("bar", 0.2, "USD/kWh")
    review = apply_source_checks(
        analysis, dict(target="2010 2011 0.300", below_target=""), reviewer="fixture"
    )
    assert review.status == "needs_review"
    assert analysis.series[0].points[0].value == 0.2
    assert review.warnings


def test_native_thousands_separators_and_unicode_minus_are_supported():
    analysis = chart("bar", -5283, "USD/kW")
    review = apply_source_checks(analysis, dict(target="2010 2011 −5 283", below_target=""))
    assert not review.warnings


def test_adjacent_thousands_objects_do_not_merge_separate_data_labels():
    from multimodal_doc_parser.visual_enrichment.source_checks import join_numeric_groups

    tokens = [
        dict(text="5", box=(10, 10, 14, 16)),
        dict(text="283", box=(16, 10, 28, 16)),
        dict(text="691", box=(40, 10, 52, 16)),
        dict(text="599", box=(58, 10, 70, 16)),
    ]
    assert join_numeric_groups(tokens) == ["5 283"]


def test_explicit_source_unit_name_normalizes_without_changing_value():
    analysis = chart("bar", 70, "gigawatt")
    review = apply_source_checks(analysis, dict(target="70", below_target="Note: GW = gigawatt."))
    assert analysis.series[0].unit == "GW"
    assert analysis.series[0].points[0].value == 70 and review.adjustments


def test_repeated_chart_conflicts_are_flagged_without_selecting_a_winner():
    from pathlib import Path

    from multimodal_doc_parser.document_schema.document import ChartElement, Document
    from multimodal_doc_parser.visual_enrichment.source_checks import flag_conflicting_chart_values

    document = Document.model_validate_json(
        Path("docs/reference/example-parsed-document.json").read_text()
    )
    for identity, values in [("first", [0.1, 0.2, 0.3, 0.4]), ("second", [0.2, 0.1, 0.3, 0.4])]:
        analysis = VisualAnalysis(
            kind="chart",
            description="A repeated line chart",
            series=[
                dict(
                    name="A",
                    unit="ms",
                    points=[
                        dict(label=str(2010 + i), value=v, value_basis="printed_label")
                        for i, v in enumerate(values)
                    ],
                )
            ],
        )
        document.elements.append(
            ChartElement(
                id=identity,
                source_regions=document.elements[0].source_regions,
                provenance=dict(producer="fixture", method="extracted"),
                asset=dict(path="fixture.png"),
                description=dict(
                    text="Repeated chart",
                    provenance=dict(producer="fixture", method="generated"),
                    analysis=analysis,
                ),
            )
        )
    flag_conflicting_chart_values(document)
    assert all(e.quality_review.status == "needs_review" for e in document.elements[-2:])
    assert all(e.quality_review.warnings for e in document.elements[-2:])
    # The guard annotates uncertainty instead of overwriting conflicting values.
    assert document.elements[-2].description.analysis.series[0].points[0].value == 0.1
    assert document.elements[-1].description.analysis.series[0].points[0].value == 0.2
