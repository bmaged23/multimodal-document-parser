"""Evidence-group recall, numeric/unit checks and persisted missing-score semantics."""

import json

import pytest

from multimodal_doc_parser.evaluation.rag_benchmark import (
    BenchmarkQuestion,
    EvidenceNeed,
    numeric_checks,
    retrieval_metrics,
    unit_checks,
)
from multimodal_doc_parser.evaluation.rag_metrics_report import export_metrics


def question():
    need = EvidenceNeed(
        document_id="sha256:" + "a" * 64,
        element_id="table",
        pages=[1],
        cells=[(1, 1)],
        accepted_child_ids=["child:a", "child:alternative"],
        accepted_parent_ids=["parent:a"],
        source_text="0.060 USD/kWh",
    )
    return BenchmarkQuestion(
        id="q001",
        category="table",
        question="What is the printed cost?",
        reference="The cost is 0.060 USD/kWh.",
        required_facts=["0.060 USD/kWh"],
        expected_abstention=False,
        evidence=[need],
        group_id="group-a",
        split="test",
        reference_status="synthetic test",
        numeric_values=["0.060"],
    )


def test_retrieval_uses_alternative_evidence_and_respects_k():
    q = question()
    ranked = dict(
        matched_children=[dict(child_id="irrelevant")] * 5 + [dict(child_id="child:alternative")]
    )
    assert retrieval_metrics(q, ranked, k=5)["reciprocal_rank"] == 0
    assert retrieval_metrics(q, ranked, k=5)["evidence_recall"] == 0
    result = retrieval_metrics(q, ranked, k=10)
    assert result["complete_evidence_success"] and result["reciprocal_rank"] == pytest.approx(1 / 6)


def test_numeric_equivalence_and_missing_units_are_separate():
    q = question()
    assert numeric_checks(q, "The cost is 0.06.")["all_expected_mentions"]
    assert not unit_checks(q, "The cost is 0.06.")["all_present"]
    assert unit_checks(q, "The cost is 0.06 US dollars per kWh.")["all_present"]
    assert not numeric_checks(q, "The cost is 0.6 USD/kWh.")["all_expected_mentions"]
    q.numeric_values = ["4443", "-46", "2011/12"]
    assert numeric_checks(q, "4,443 GW; −46%; base year 2011/12.")["all_expected_mentions"]


def test_unanswerable_retrieval_has_no_positive_gold_score():
    q = question()
    q.category = "unanswerable"
    q.expected_abstention = True
    q.evidence = []
    assert retrieval_metrics(q, dict(matched_children=[]))["applicable"] is False


def test_report_saves_per_metric_outputs_and_does_not_turn_missing_scores_into_zero(tmp_path):
    for i, status, value in [(1, "scored", 1.0), (2, "unavailable", None)]:
        path = tmp_path / "cases" / f"q{i:03d}"
        path.mkdir(parents=True)
        (path / "result.json").write_text(
            json.dumps(
                dict(
                    id=f"q{i:03d}",
                    category="table",
                    split="test",
                    status="completed" if i == 1 else "blocked_by_account_access",
                    retrieval_metrics={},
                    scenarios=dict(
                        agent=dict(
                            status="completed" if i == 1 else "unavailable",
                            metrics=dict(faithfulness=dict(status=status, value=value)),
                        )
                    ),
                )
            )
        )
    result = export_metrics(tmp_path, 100)
    metric = next(a for a in result["aggregates"] if a["split"] == "all" and a["category"] == "all")
    assert metric["mean"] == 1.0 and metric["scored"] == 1 and metric["observed"] == 2
    assert result["expected_questions"] == 100 and result["completed_questions"] == 1
    assert (tmp_path / "metrics.csv").exists() and (tmp_path / "metrics.jsonl").exists()
    assert "NaN" not in (tmp_path / "summary.json").read_text()
    assert len(json.loads((tmp_path / "failures.json").read_text())) == 1


def test_question_id_cannot_escape_case_directory():
    q = question().model_dump()
    q["id"] = "../../.env"
    with pytest.raises(ValueError):
        BenchmarkQuestion.model_validate(q)
