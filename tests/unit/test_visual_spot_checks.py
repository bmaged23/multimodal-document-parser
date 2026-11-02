"""Spot checks must expose missing units and mismatched sources rather than overclaim."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_numeric_match_without_required_unit_fails(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "visual_checks", "scripts/evaluate_visual_enrichment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = {
        "document_id": "fixture",
        "visuals": [
            {
                "element_id": "chart",
                "analysis": {
                    "series": [
                        {
                            "unit": None,
                            "points": [
                                {"label": "35%", "value": 18.4, "value_basis": "printed_label"}
                            ],
                        }
                    ]
                },
            }
        ],
    }
    (tmp_path / "enrichment-manifest.json").write_text(json.dumps(manifest))
    reference = {
        "document_id": "fixture",
        "annotation_provenance": "synthetic fixture",
        "chart_points": [{"element_id": "chart", "label": "35%", "value": 18.4, "unit": "ms"}],
    }
    assert module.evaluate(tmp_path, reference)["passed"] == 0
    reference["document_id"] = "other"
    with pytest.raises(ValueError, match="different source"):
        module.evaluate(tmp_path, reference)


def test_wrong_raster_type_is_not_hidden_by_correct_numeric_checks(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "visual_checks", "scripts/evaluate_visual_enrichment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "enrichment-manifest.json").write_text(
        json.dumps(
            {
                "document_id": "fixture",
                "visuals": [{"element_id": "raster", "analysis": {"kind": "flowchart"}}],
            }
        )
    )
    result = module.evaluate(
        tmp_path,
        {
            "document_id": "fixture",
            "annotation_provenance": "synthetic fixture",
            "visual_types": [{"element_id": "raster", "kind": "image"}],
        },
    )
    assert result["checked"] == 1 and result["passed"] == 0
