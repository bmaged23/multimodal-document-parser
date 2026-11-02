"""Report bounded source-cell coverage independently of any LLM judge."""

import argparse
import json
from collections import defaultdict
from html import escape
from pathlib import Path
from statistics import mean

from multimodal_doc_parser.evaluation.chunking_baselines import (
    FixedRetriever,
    cell_aware_source_coverage,
)
from multimodal_doc_parser.evaluation.rag_benchmark import BenchmarkQuestion, save
from multimodal_doc_parser.rag_preparation.retrieval import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    from compare_rag_approaches import VARIANTS

    questions = [
        BenchmarkQuestion.model_validate_json(line)
        for line in Path("outputs/rag_evaluation_datasets/ragas-100-v1-reviewed/questions.jsonl")
        .read_text()
        .splitlines()
    ]
    adaptive = Retriever(Path("outputs/rag_indexes/bge-large-hybrid-02"))
    fixed = FixedRetriever(args.directory / "fixed-index")
    groups = defaultdict(list)
    rows = []
    for variant, options in VARIANTS.items():
        retriever = fixed if options["fixed"] else adaptive
        for q in questions:
            root = args.directory / "variants" / variant / "cases" / q.id
            if options.get("reuse"):
                original = Path("outputs/quality_evaluations/ragas-100-v1/full-01/cases")
                contexts = json.loads((original / q.id / "agent/answer.json").read_text())[
                    "evidence"
                ]["contexts"]
            else:
                contexts = json.loads((root / "context.json").read_text())["contexts"]
            coverage = cell_aware_source_coverage(q, contexts, retriever.parents)
            tokens = retriever.budget.count("\n\n".join(b["text"] for b in contexts))
            assert tokens <= 2400
            row = dict(
                variant=variant,
                id=q.id,
                category=q.category,
                coverage=coverage,
                context_tokens=tokens,
            )
            rows.append(row)
            for category in ["all", q.category]:
                groups[(variant, category)].append(row)
    aggregates = []
    for (variant, category), items in sorted(groups.items()):
        scored = [r for r in items if r["coverage"]["applicable"]]
        aggregates.append(
            dict(
                variant=variant,
                category=category,
                source_evidence_recall=mean(r["coverage"]["recall"] for r in scored)
                if scored
                else None,
                complete_source_evidence=mean(float(r["coverage"]["complete"]) for r in scored)
                if scored
                else None,
                scored=len(scored),
                average_context_tokens=mean(r["context_tokens"] for r in items),
            )
        )
    save(
        args.directory / "bounded-retrieval-comparison.json",
        dict(
            method="source-cell-coverage-v2, within the actual 2400-token answer context",
            scope=(
                "No judge calls. Initial contexts for plain RAG; final saved production "
                "context after any rewrite for the agent."
            ),
            limitations=(
                "Source-bound silver evidence. Exact full column-label/value segments required "
                "for cells; full saved parts required for narrative and chart points. "
                "Conservative source inclusion is not proof that the LLM can interpret the "
                "context or answer correctly. Same previously inspected 100 questions."
            ),
            rows=rows,
            aggregates=aggregates,
        ),
    )
    table = "".join(
        f"<tr><td>{escape(r['variant'])}</td><td>{r['complete_source_evidence']:.3f}</td>"
        f"<td>{r['source_evidence_recall']:.3f}</td><td>{r['scored']}</td>"
        f"<td>{r['average_context_tokens']:.0f}</td></tr>"
        for r in aggregates
        if r["category"] == "all"
    )
    (args.directory / "retrieval-report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Retrieval comparison without an LLM judge</title><style>"
        "body{font:16px system-ui;margin:30px;color:#172538}"
        "td,th{padding:10px;border:1px solid #ccd;text-align:left}"
        "table{border-collapse:collapse}</style><h1>Retrieval comparison</h1>"
        "<p>Same source representations, 100 questions, BGE-large 1024-dimensional "
        "GPU embeddings, top 5 results and a 2400-token context ceiling. "
        "Scores below use source coordinates and text spans, not an LLM judge.</p>"
        "<p><b>Complete evidence:</b> fraction of the 90 answerable questions with "
        "all expected evidence present. <b>Evidence recall:</b> average fraction "
        "of required evidence groups present. Partial table rows can earn credit "
        "when all requested column-label/value segments are present. Narrative "
        "paragraphs and chart points require full saved parts. These are conservative "
        "source-inclusion checks, not answer-accuracy scores.</p><p>Agent uses its "
        "saved final context after any rewrite; other variants use one retrieval. "
        "This is an exploratory comparison on previously inspected questions.</p>"
        "<table><tr><th>Approach</th><th>Complete evidence</th><th>Evidence recall</th>"
        "<th>Scored</th><th>Average context tokens</th></tr>" + table + "</table>"
        '<p><a href="bounded-retrieval-comparison.json">Per-question/category details</a> · '
        '<a href="report.html">Answer-quality comparison progress</a></p></html>'
    )
    for row in aggregates:
        if row["category"] == "all":
            print(json.dumps(row))


if __name__ == "__main__":
    main()
