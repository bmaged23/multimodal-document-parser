"""Preserve GPT-OSS reference drafts separately from source-compiled fact targets."""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from multimodal_doc_parser.evaluation.rag_benchmark import (
    BenchmarkQuestion,
    digest,
    save,
    validate_dataset,
)
from multimodal_doc_parser.evaluation.ragas_cloud_judge import OllamaRagasJudge


class ReferenceDraft(BaseModel):
    id: str
    answer: str


class ReferenceBatch(BaseModel):
    items: list[ReferenceDraft]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if (args.output_directory / "questions.jsonl").exists():
        raise ValueError("Completed dataset already exists; do not overwrite it")
    questions = [
        BenchmarkQuestion.model_validate_json(line)
        for line in (args.source / "questions.jsonl").read_text().splitlines()
    ]
    for start in range(0, len(questions), 5):
        batch = questions[start : start + 5]
        output = args.output_directory / f"batch-{start // 5 + 1:02d}.json"
        if output.exists():
            result = ReferenceBatch.model_validate_json(output.read_text())
        else:
            judge = OllamaRagasJudge(
                args.output_directory / "raw_reference_generation" / f"batch-{start // 5 + 1:02d}"
            )
            packets = [
                dict(
                    id=q.id,
                    question=q.question,
                    source_evidence=[e.source_text for e in q.evidence],
                    source_checked_fact_targets=q.required_facts,
                    expected_abstention=q.expected_abstention,
                )
                for q in batch
            ]
            prompt = (
                "Draft concise reference answers for these source-bound benchmark "
                "questions. Use only the supplied evidence and fact targets. "
                "Preserve all requested numbers, years, units, code labels and "
                "qualifications. Do not add inferred motion, identities, locations "
                "or precise scatter coordinates. For expected_abstention explain "
                "what information is missing. Return each id once. Do not write "
                "evaluation scores.\n"
            ) + json.dumps(packets, ensure_ascii=False)
            result = judge.generate(prompt, ReferenceBatch)
            if {r.id for r in result.items} != {q.id for q in batch} or len(result.items) != len(
                batch
            ):
                raise ValueError("Reference response IDs differ from the requested batch")
            save(output, result.model_dump())
        drafts = {r.id: r.answer for r in result.items}
        for q in batch:
            q.reference_draft = drafts[q.id]
            q.reference_generator_model = "gpt-oss:120b"
            q.reference_status += (
                "; separate GPT-OSS draft retained; source-compiled facts remain scoring reference"
            )
        save(
            args.output_directory / "progress.json",
            dict(status="generating_reference_drafts", completed=start + len(batch), total=100),
        )
        print(f"Reference drafts {start + len(batch)}/100", flush=True)
    validate_dataset(questions)
    target = args.output_directory / "questions.jsonl"
    target.write_text("".join(q.model_dump_json() + "\n" for q in questions))
    save(
        args.output_directory / "expected-evidence.json",
        {q.id: [e.model_dump() for e in q.evidence] for q in questions},
    )
    save(
        args.output_directory / "manifest.json",
        dict(
            benchmark_id="ragas-100-v1",
            questions_sha256=digest(target),
            source_dataset_manifest_sha256=digest(args.source / "manifest.json"),
            validation=validate_dataset(questions),
            reference_generator_model="gpt-oss:120b",
            judge_model="gpt-oss:120b",
            reference_policy=(
                "Draft answers from GPT-OSS are saved separately; source-compiled "
                "reviewed fact targets remain the evaluation reference to reduce "
                "circular self-confirmation. Single-assistant silver references, no "
                "independent human gold claim."
            ),
        ),
    )
    save(args.output_directory / "progress.json", dict(status="complete", completed=100, total=100))


if __name__ == "__main__":
    main()
