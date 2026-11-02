"""Record an assistant review of 20 saved cases, without re-parsing or re-answering."""

import json
from html import escape
from pathlib import Path

from multimodal_doc_parser.evaluation.rag_benchmark import digest, save

NOTES = {
    "q001": (
        "judge_reference_error",
        (
            "Answer covers both risks. The judge turned the reference list into an "
            "invented relationship: manufacturing dynamics are a factor in tariffs."
            " The zero factual score is misleading."
        ),
    ),
    "q009": (
        "metric_error_and_extra_detail",
        (
            "Answer includes the three required mechanisms plus additional "
            "battery-cost detail from retrieved material. Faithfulness failed to "
            "score; citation support scored 1. Extra supported detail should not "
            "automatically be treated as false."
        ),
    ),
    "q012": (
        "citation_granularity",
        (
            "Answer matches the reference, but its cited child contains "
            "introductory data-convention text rather than the averaging rule "
            "supplied in expanded context. The citation-support failure is "
            "meaningful."
        ),
    ),
    "q018": (
        "judge_scope_sensitivity",
        (
            "Answer expresses both the unusually high uncertainty and paused IMF "
            "engagement over recognition. The 0.5 factual score understates this "
            "semantic match; inspect entailment directions and claim scope."
        ),
    ),
    "q021": (
        "judge_scope_sensitivity",
        (
            "All four AX-17 quantities and associations match the source cells. The"
            " reference omits the sensor identity, which is supplied by the "
            "question. Its zero factual score must not be treated as a wrong "
            "answer."
        ),
    ),
    "q029": (
        "judge_scope_and_table_headers",
        (
            "The values match the Onshore-wind row. Reference claim decomposition "
            "omits technology scope. Citations include the row and some headers, "
            "but generic column names and an absent year-header child make citation"
            " interpretation incomplete."
        ),
    ),
    "q041": (
        "supported_match",
        (
            "Currency, base year and SNA system match the Algeria cells. "
            "Correctness and citation support both score 1."
        ),
    ),
    "q049": (
        "supported_match_with_extra_detail",
        (
            "Answer preserves exclusion of CO2, price per tonne, avoided volume and"
            " climate benefit. Attribution to the price commission is supported by "
            "the footnote; a concise reference need not enumerate it."
        ),
    ),
    "q051": (
        "supported_match",
        (
            "Both Solar-PV endpoint values, units and years match the saved chart "
            "evidence and appear in the cited child."
        ),
    ),
    "q057": (
        "judge_scope_and_citation_granularity",
        (
            "Hydropower values and years match the reference. The reference omits "
            "the series name; its zero correctness is misleading. The cited child "
            "has Solar-PV/CSP content rather than Hydropower, explaining the "
            "citation-support failure."
        ),
    ),
    "q062": (
        "citation_granularity",
        (
            "The three India values match the reference, but the cited child "
            "contains World points. Parent expansion supplied additional series; a "
            "valid child ID is insufficient support."
        ),
    ),
    "q065": (
        "supported_uncertainty",
        (
            "The answer avoids inventing exact scatter-point values. The selected "
            "reference/source excerpt is not a full visual verification; retain "
            "uncertainty about exact-value availability."
        ),
    ),
    "q066": (
        "supported_core_extra_visual_detail",
        (
            "Answer includes the turbine, red markings and solar panels. Extra "
            "blade-count, colour and mounting details are supported by generated "
            "descriptions, not independently verified here. F1 penalizes details "
            "absent from the short reference."
        ),
    ),
    "q068": (
        "visual_motion_overclaim",
        (
            "The answer says a component is being lowered. A still image does not "
            "establish that direction. This is a real visual-enrichment error "
            "reproduced faithfully by the answer, so faithfulness 1 does not prove "
            "visual truth."
        ),
    ),
    "q070": (
        "partial_reference_and_extra_visual_detail",
        (
            "Answer covers dam, spillways, water, spray, orange structures and "
            "buildings but omits mountainous surroundings. Extra building details "
            "require visual verification. A zero score overstates the mismatch."
        ),
    ),
    "q071": (
        "supported_cross_page_match",
        (
            "Afghan-afghani currency and the government-finance manual code 2001 "
            "match their separate Table-G page groups and the cited chunks."
        ),
    ),
    "q082": (
        "retrieval_and_citation_failure",
        (
            "The final context lacks the 2024 actual-outturn caveat, so the answer "
            "is incomplete. It gives the correct national-accounts year 2020 but "
            "cites a government-finance child reporting 2023, not the supporting "
            "national-accounts row."
        ),
    ),
    "q090": (
        "derived_arithmetic_undercredit",
        (
            "179.8 / 467 * 100 rounds to 38.5%. The amounts and calculation match "
            "the source and reference. Faithfulness can undercredit a valid derived"
            " percentage that is not printed verbatim."
        ),
    ),
    "q091": (
        "appropriate_refusal",
        (
            "The agent declines an unsupported 2035 exact projection. This is a "
            "refusal-decision check, not a full factuality assessment of arbitrary "
            "refusal text."
        ),
    ),
    "q100": (
        "appropriate_refusal",
        (
            "The agent declines to invent twelve raw observations from summary "
            "statistics. The saved data cannot determine those observations."
        ),
    ),
}


def main():
    baseline = Path("outputs/quality_evaluations/ragas-100-v1/full-01")
    dataset = Path("outputs/rag_evaluation_datasets/ragas-100-v1-reviewed/questions.jsonl")
    output = Path("outputs/quality_evaluations/ragas-100-v1/audit-20-v1")
    questions = {q["id"]: q for q in map(json.loads, dataset.read_text().splitlines())}
    records = []
    for identity, (finding, note) in NOTES.items():
        path = baseline / "cases" / identity / "result.json"
        row = json.loads(path.read_text())
        agent = row["scenarios"]["agent"]
        records.append(
            dict(
                id=identity,
                category=row["category"],
                question=row["question"],
                reference=row["reference"],
                answer=agent["answer"],
                original_metrics=agent["metrics"],
                evidence=questions[identity]["evidence"],
                finding=finding,
                review_note=note,
                saved_result_sha256=digest(path),
            )
        )
    save(
        output / "audit.json",
        dict(
            reviewer="Codex assistant; not an independent human reviewer",
            method=(
                "Purposive 20-case text/source-coordinate audit, including metric "
                "disagreement, citations, images, cross-page joins and refusals. Not a "
                "random sample or prevalence estimate. No parsing or answer generation "
                "rerun."
            ),
            dataset_sha256=digest(dataset),
            cases=records,
            limitations=(
                "Uses saved source text and source-bound silver references. No new "
                "visual PDF adjudication; image-detail judgments remain limited. Do not"
                " extrapolate rates to the full dataset."
            ),
        ),
    )
    articles = "".join(
        f"<article><h2>{r['id']} — {escape(r['finding'])}</h2>"
        f"<p><b>Question:</b> {escape(r['question'])}</p>"
        f"<p><b>Reference:</b> {escape(r['reference'])}</p>"
        f"<p><b>Answer:</b> {escape(r['answer']['answer'])}</p>"
        f"<p><b>Review:</b> {escape(r['review_note'])}</p></article>"
        for r in records
    )
    (output / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>20-case evaluation audit</title><style>body{font:16px system-ui;"
        "max-width:1100px;margin:30px auto;padding:15px;color:#172538}"
        "article{border-top:1px solid #ccd;padding:12px 0}p{white-space:pre-wrap}</style>"
        "<h1>20-case evaluation audit</h1><p>Assistant review of saved text and source "
        "coordinates; not independent human ground truth or a random sample. "
        "No parsing or answer generation rerun. Visual details were not independently "
        "rechecked against the PDFs in this audit.</p>" + articles + "</html>"
    )
    print(f"Saved {len(records)} reviewed cases to {output}")


if __name__ == "__main__":
    main()
