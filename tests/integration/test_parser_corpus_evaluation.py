"""Verify source scoring cannot hide failures or confuse repetition with recovery."""

import importlib.util
import json
import random
from pathlib import Path

import pytest


@pytest.fixture
def scorer():
    path = Path("scripts/parser_evaluation/evaluate_parser_corpus.py")
    spec = importlib.util.spec_from_file_location("corpus_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bit_vector_distance_matches_dynamic_programming(scorer):
    def reference(a, b):
        row = list(range(len(b) + 1))
        for i, x in enumerate(a):
            current = [i + 1]
            for j, y in enumerate(b):
                current.append(min(current[-1] + 1, row[j + 1] + 1, row[j] + (x != y)))
            row = current
        return row[-1]

    rng = random.Random(17)
    for _ in range(100):
        a = [rng.randrange(4) for _ in range(rng.randrange(30))]
        b = [rng.randrange(4) for _ in range(rng.randrange(30))]
        assert scorer.distance(a, b) == reference(a, b)


def test_extra_repetition_reduces_precision(scorer):
    row = scorer.text_counts("mission to mars", "mission to mars mission to mars")
    result = scorer.summarize_text([row])
    assert result["recall"] == 1
    assert result["precision"] == 0.5
    assert result["f1"] < 1


def test_one_visual_prediction_cannot_match_two_source_regions(scorer):
    ref = {"category": "photo", "bbox": [0, 0, 1, 1]}
    prediction = {"label": "image", "bbox": [0, 0, 1, 1]}
    assert len(scorer.maximum_matches([ref, ref], [prediction], 0.5)) == 1


def test_table_values_elsewhere_on_page_cannot_earn_chart_credit(scorer):
    ref = {"id": "chart", "region": [0.1, 0.5, 0.8, 0.8], "category": "35%", "value": "18.4"}
    page = {
        "blocks": [
            {
                "label": "table",
                "bbox": [0.1, 0.1, 0.8, 0.3],
                "text": "<table><tr><td>35%</td><td>18.4</td></tr></table>",
            }
        ]
    }
    assert not scorer.chart_checks(page, [ref])[0]["passed"]
    page["blocks"][0]["bbox"] = ref["region"]
    assert scorer.chart_checks(page, [ref])[0]["passed"]


def test_budget_values_must_have_correct_objective_and_column(scorer):
    ref = {"values": {"1.1": [1974, 1740]}, "group_rowspans": {}}
    page = {
        "blocks": [
            {"text": "<table><tr><td>1.1 Earth</td><td>1,740</td><td>1,974</td></tr></table>"}
        ]
    }
    rows, _ = scorer.check_nasa_table(page, ref)
    assert not any(r["passed"] for r in rows)
    page["blocks"][0]["text"] = (
        "<table><tr><td>1.1 Earth</td><td>1,974</td><td>1,740</td></tr></table>"
    )
    rows, _ = scorer.check_nasa_table(page, ref)
    assert all(r["passed"] for r in rows)


def test_truncated_generation_is_not_scored_as_completed(scorer, tmp_path):
    scorer.ROOT = tmp_path
    run = tmp_path / "run"
    run.mkdir()
    (run / "page_0001.json").write_text(
        json.dumps(
            {
                "finish_reason": "length",
                "text": "<PAGE><BLOCK>text[0,0,1000,1000]<FORK>apparently correct partial text",
            }
        )
    )
    result = scorer.load_page(run, 1, 2)
    assert not result["complete"]
    assert result["text"] == ""
    assert result["blocks"] == []
