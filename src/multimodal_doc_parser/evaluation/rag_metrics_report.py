"""Persist finite per-case scores, explicit missingness and scoped aggregates."""

import csv
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any

from multimodal_doc_parser.evaluation.rag_benchmark import save

METRIC_MEANINGS = {
    "abstention_correct": (
        "Did the model correctly choose whether to answer or flag insufficient evidence? "
        "This checks that decision, not whether its answer is correct."
    ),
    "answer_relevancy": (
        "How closely does the answer address the question? A relevant answer can still be wrong."
    ),
    "citation_ids_valid": (
        "Does the answer include citations, with every citation ID belonging to the supplied "
        "evidence? This checks IDs, not whether those sources support the claims."
    ),
    "citation_support": (
        "How much of the answer is supported by the specific child chunks it cites, "
        "rather than by the wider context? Assessed by the LLM judge."
    ),
    "context_precision": (
        "Did retrieval put useful evidence ahead of irrelevant material? "
        "The LLM judge assesses usefulness against the reference answer."
    ),
    "context_recall": (
        "How much of the reference answer is supported by the context supplied to the model? "
        "Assessed by the LLM judge."
    ),
    "expanded_evidence_recall": (
        "What fraction of the expected source evidence was included in the agent's context "
        "after adding surrounding parent content? Checked against saved source locations."
    ),
    "factual_correctness": (
        "How well do the answer's factual claims match the reference answer? "
        "The judge's F1 score accounts for missing and extra claims; it is not the "
        "percentage of questions answered correctly."
    ),
    "faithfulness": (
        "What fraction of the answer's claims are supported by the supplied context? "
        "This does not verify that the parsed context itself is correct or the answer complete."
    ),
    "numeric_mentions": (
        "Does the answer contain all expected numbers, allowing equivalent number formats? "
        "This does not check whether each number is linked to the correct fact."
    ),
    "unit_mentions": (
        "Does the answer mention all expected units? This does not check whether each unit "
        "is attached to the correct number."
    ),
}


def metric_meaning(metric: str) -> str:
    if metric in METRIC_MEANINGS:
        return METRIC_MEANINGS[metric]
    prefix, separator, k = metric.rpartition("_at_")
    if separator and k.isdigit():
        meanings = {
            "evidence_recall": (
                f"What fraction of the required evidence groups was found among the first {k} "
                "retrieved chunks? Checked against the saved expected evidence."
            ),
            "complete_evidence_success": (
                f"For what fraction of answerable questions did the first {k} retrieved "
                "chunks contain all required evidence groups?"
            ),
            "reciprocal_rank": (
                f"How early did the first expected evidence appear within the top {k} chunks? "
                "Rank 1 earns 1, rank 2 earns 0.5, rank 3 earns 0.333; missing evidence earns 0."
            ),
            "expanded_evidence_recall": (
                f"After expanding the top {k} results with surrounding parent content, "
                "what fraction of the expected source evidence was actually included "
                "within the context budget?"
            ),
        }
        if prefix in meanings:
            return meanings[prefix]
    return "No explanation is available for this metric."


def export_metrics(directory: Path, expected_questions: int = 100) -> dict[str, Any]:
    records = [
        json.loads(p.read_text()) for p in sorted((directory / "cases").glob("*/result.json"))
    ]
    (directory / "metrics.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records)
    )
    flat = []
    for row in records:
        for metric, value in row.get("retrieval_metrics", {}).items():
            flat.append(
                dict(
                    question_id=row["id"],
                    category=row["category"],
                    split=row["split"],
                    scenario="retrieval",
                    metric=metric,
                    value=value.get("value"),
                    status=value["status"],
                )
            )
        for scenario, data in row.get("scenarios", {}).items():
            for metric, value in data.get("metrics", {}).items():
                flat.append(
                    dict(
                        question_id=row["id"],
                        category=row["category"],
                        split=row["split"],
                        scenario=scenario,
                        metric=metric,
                        value=value.get("value"),
                        status=value["status"],
                    )
                )
    with (directory / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "question_id",
                "category",
                "split",
                "scenario",
                "metric",
                "value",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(flat)
    aggregates: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in flat:
        for split in ["all", row["split"]]:
            for category in ["all", row["category"]]:
                aggregates[(split, category, row["scenario"], row["metric"])].append(row)
    summary: list[dict[str, Any]] = []
    for (split, category, scenario, metric), rows in sorted(aggregates.items()):
        scored = [r["value"] for r in rows if r["status"] == "scored" and r["value"] is not None]
        summary.append(
            dict(
                split=split,
                category=category,
                scenario=scenario,
                metric=metric,
                mean=mean(scored) if scored else None,
                median=median(scored) if scored else None,
                scored=len(scored),
                observed=len(rows),
                statuses=dict(Counter(r["status"] for r in rows)),
            )
        )
    result = dict(
        expected_questions=expected_questions,
        recorded_questions=len(records),
        completed_questions=sum(r.get("status") == "completed" for r in records),
        question_statuses=dict(Counter(r.get("status", "unknown") for r in records)),
        aggregates=summary,
        interpretation=(
            "Means include only finite scored metrics. Missing, failed and "
            "inapplicable scores are not zeros. Read status counts and "
            "completion rates alongside means. Development and frozen-test "
            "results are separate."
        ),
    )
    save(directory / "summary.json", result)
    failures = [
        dict(question_id=r["id"], scenario=s, error=v.get("error"), status=v.get("status"))
        for r in records
        for s, v in r.get("scenarios", {}).items()
        if v.get("status") not in ["completed", "not_applicable"]
    ]
    save(directory / "failures.json", failures)
    rows_html = "".join(
        "<tr>"
        + "".join(
            f"<td>{escape(str(v))}</td>"
            for v in [
                r["id"],
                r["category"],
                r["split"],
                r.get("status"),
                "; ".join(f"{s}: {v.get('status')}" for s, v in r.get("scenarios", {}).items()),
            ]
        )
        + "</tr>"
        for r in records
    )
    table = "".join(
        "<tr>"
        + "".join(
            f"<td>{escape(str(v))}</td>"
            for v in [
                a["scenario"],
                a["metric"],
                round(a["mean"], 4) if a["mean"] is not None else "unavailable",
                a["scored"],
                a["statuses"],
                metric_meaning(a["metric"]),
            ]
        )
        + "</tr>"
        for a in summary
        if a["split"] == "all" and a["category"] == "all"
    )
    notes_path = directory / "evaluation-notes.json"
    note_html = ""
    if notes_path.exists():
        notes = json.loads(notes_path.read_text())
        note_html = f"<aside><strong>Evaluation audit:</strong> {escape(notes['message'])}</aside>"
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>100-question Ragas evaluation</title><style>
body{{font:15px system-ui;margin:24px;color:#172538}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}
td,th{{border:1px solid #ccd;padding:8px;text-align:left}}
.metric-meaning{{min-width:300px;max-width:600px}}
td{{vertical-align:top}}
aside{{background:#fff4d8;padding:12px}}</style>
<h1>100-question Ragas evaluation</h1><aside>{escape(result["interpretation"])}</aside>
{note_html}
<p>{result["recorded_questions"]}/{expected_questions} recorded;
{result["completed_questions"]} completed.</p>
<p>Judge: GPT-OSS 120B on Ollama Cloud. Agent: Gemma 4 31B.
Source-bound silver references; not independent human ground truth.</p>
<p><a href="metrics.csv">Per-metric CSV</a> ·
<a href="metrics.jsonl">Per-question JSONL</a> ·
<a href="summary.json">Aggregates by split/category</a> ·
<a href="failures.json">Failures</a> · <a href="configuration.json">Configuration</a></p>
<h2>Aggregate scores</h2>
<p>Scores range from 0 to 1; higher is better. Mean is the average of scored cases,
not necessarily the fraction of correct answers. Scored is the number included in that average.
N/A and errors are excluded. Agent uses the actual pipeline; oracle receives expected
evidence directly, bypassing retrieval.</p>
<table><tr><th>Scenario</th><th>Metric</th><th>Mean</th>
<th>Scored</th><th>Statuses</th><th class="metric-meaning">Meaning</th></tr>{table}</table>
<h2>Question status</h2><table><tr><th>ID</th><th>Category</th><th>Split</th>
<th>Status</th><th>Scenarios</th></tr>{rows_html}</table></html>"""
    (directory / "report.html").write_text(html)
    return result
