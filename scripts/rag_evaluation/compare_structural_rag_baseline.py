"""Compare practical structural plain RAG with the existing adaptive/agent results."""

import argparse
import gc
import json
import time
import urllib.error
from pathlib import Path

from multimodal_doc_parser.evaluation.chunking_baselines import cell_aware_source_coverage
from multimodal_doc_parser.evaluation.contextual_judge_v2 import assess_answer
from multimodal_doc_parser.evaluation.rag_benchmark import (
    BenchmarkQuestion,
    digest,
    numeric_checks,
    save,
    unit_checks,
    validate_dataset,
)
from multimodal_doc_parser.evaluation.ragas_cloud_judge import OllamaRagasJudge
from multimodal_doc_parser.evaluation.semantic_structure_baseline import build_semantic_bundles
from multimodal_doc_parser.evaluation.simple_structure_baseline import (
    SimpleStructureRetriever,
)
from multimodal_doc_parser.pipeline_services.settings import ServiceSettings
from multimodal_doc_parser.pipeline_services.stage_handlers import gpu_lease
from multimodal_doc_parser.rag_preparation.gpu_embeddings import GPUEmbeddings
from multimodal_doc_parser.rag_preparation.retrieval import build_index

NEW_VARIANTS = {"simple-structure-dense": "dense", "simple-structure-hybrid": "hybrid"}
REUSED_VARIANTS = ["adaptive-hybrid-plain", "adaptive-parent-plain", "adaptive-parent-agent"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--prior-comparison",
        type=Path,
        default=Path("outputs/quality_evaluations/rag-approach-comparison-v1"),
    )
    args = parser.parse_args()
    import compare_rag_approaches as shared
    from run_ragas_benchmark import COLLECTIONS, metric_score

    shared.VARIANTS = {name: {} for name in [*NEW_VARIANTS, *REUSED_VARIANTS]}
    prior = args.prior_comparison
    dataset = Path("outputs/rag_evaluation_datasets/ragas-100-v1-reviewed/questions.jsonl")
    index = Path("outputs/rag_indexes/bge-large-hybrid-02")
    calibration = Path("outputs/quality_evaluations/judge-validation-v2/summary.json")
    gate = json.loads(calibration.read_text())
    if not gate["contextual_gate_passed"]:
        raise ValueError("Judge validation gate is not passed")
    questions = [BenchmarkQuestion.model_validate_json(s) for s in dataset.read_text().splitlines()]
    validate_dataset(questions)
    config = dict(
        algorithm="simple-structure-v2",
        dataset_sha256=digest(dataset),
        source_index_manifest_sha256=digest(index / "index_manifest.json"),
        judge_validation_sha256=digest(calibration),
        judge="gpt-oss:120b",
        answer_model="gemma4:31b",
        embedding_dimensions=1024,
        child_target_tokens=350,
        child_soft_maximum_tokens=500,
        repeated_explanation_budget_tokens=100,
        top_k=5,
        context_tokens=2400,
        variants=shared.VARIANTS,
        chunk_policy=(
            "Contiguous text within a source section; repeat full heading hierarchy "
            "on every continuation. Separate physical tables with repeated caption "
            "and headers; separate charts/images. Split only at whole paragraph, sentence, "
            "table-row, chart-series or chart-point boundaries. Preserve and flag indivisible "
            "oversized units. Include uniquely related explanations when they fit; otherwise "
            "keep them independently searchable under their item scope. "
            "No parent expansion or rewrite."
        ),
        representation_difference=(
            "The new baseline treats blank stubs in an otherwise "
            "annotated header band as inferred header structure, so actual header paths repeat. "
            "Footnotes are searchable table-associated content rather than repeated in every "
            "row group. Existing production parsing/chunking and saved results are unchanged. "
            "This is a practical baseline comparison, not a pure boundary-only ablation."
        ),
        reuse_policy="Reuse saved advanced answers and scores; no production reruns.",
        prior_configuration_sha256=digest(prior / "configuration.json"),
        interpretation="Same previously inspected 100 questions; exploratory, not an unseen test.",
    )
    path = args.output_directory / "configuration.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Use a fresh directory for changed comparison settings")
    save(path, config)
    save(args.output_directory / "judge-validation.json", gate)

    def report():
        shared.report(args.output_directory, questions)
        path = args.output_directory / "summary.json"
        data = json.loads(path.read_text())
        data["limitations"] = config["interpretation"] + " " + config["representation_difference"]
        data["primary_baseline"] = "simple-structure-hybrid"
        data["fixed_size_results"] = (
            "Retained separately as diagnostic baselines, not the headline comparison."
        )
        save(path, data)

    def import_advanced():
        for variant in REUSED_VARIANTS:
            for q in questions:
                source = prior / "variants" / variant / "cases" / q.id / "result.json"
                if not source.exists():
                    continue
                record = json.loads(source.read_text())
                if record["status"] != "completed":
                    continue
                record["reused_from"] = str(source)
                record["reused_result_sha256"] = digest(source)
                save(
                    args.output_directory / "variants" / variant / "cases" / q.id / "result.json",
                    record,
                )

    metadata = json.loads((index / "index_manifest.json").read_text())
    sources = [(index / s["bundle"]).resolve() for s in metadata["source_bundles"]]
    settings = ServiceSettings.from_environment()
    simple_index = args.output_directory / "simple-index"
    with gpu_lease(settings):
        model = GPUEmbeddings(settings.embedding_cache, revision=metadata["embedding"]["revision"])
        if not simple_index.exists():
            bundles = build_semantic_bundles(sources, args.output_directory / "simple-bundles")
            build_index(bundles, simple_index, embedder=model, batch_size=2)
        retriever = SimpleStructureRetriever(simple_index, model)
        save(args.output_directory / "gpu.json", model.metadata())
        for variant, mode in NEW_VARIANTS.items():
            for number, q in enumerate(questions, 1):
                root = args.output_directory / "variants" / variant / "cases" / q.id
                retrieval_path = root / "retrieval.json"
                if retrieval_path.exists():
                    retrieval = json.loads(retrieval_path.read_text())["result"]
                else:
                    start = time.monotonic()
                    retrieval = retriever.search(
                        q.question, top_k=10, context_tokens=2400, mode=mode, expand=False
                    )
                    save(retrieval_path, dict(result=retrieval, seconds=time.monotonic() - start))
                result_path = root / "result.json"
                if not result_path.exists():
                    ids = [h["child_id"] for h in retrieval["matched_children"][:5]]
                    contexts = retriever.expand(ids, 2400, False)
                    assert all(
                        block["context_part_ids"]
                        == retriever.by_id[block["matched_child_ids"][0]].context_part_ids
                        for block in contexts
                    )
                    scores = shared.retrieval_scores(q, retrieval, retriever)
                    coverage = cell_aware_source_coverage(q, contexts, retriever.parents)
                    scores["context_cell_evidence_recall"] = coverage.get("recall")
                    scores["context_cell_complete"] = (
                        float(coverage["complete"]) if coverage["applicable"] else None
                    )
                    save(root / "context.json", dict(contexts=contexts))
                    save(
                        result_path,
                        dict(
                            id=q.id,
                            variant=variant,
                            category=q.category,
                            question=q.question,
                            status="retrieval_complete",
                            metrics=scores,
                        ),
                    )
                if number % 20 == 0:
                    save(
                        args.output_directory / "status.json",
                        dict(
                            status="retrieval_running", variant=variant, position=number, total=100
                        ),
                    )
                    print(f"{variant} retrieval {number}/100", flush=True)
        retriever.embedder = None
        del model
        gc.collect()
    import_advanced()
    report()
    if args.retrieval_only:
        save(
            args.output_directory / "status.json",
            dict(status="retrieval_completed", new_retrieval_cases=200),
        )
        return

    # Do not compete with the existing cloud batch. Read-only polling preserves
    # its work and imports completed advanced cases without regenerating them.
    while True:
        state = json.loads((prior / "status.json").read_text())
        if state["status"] == "completed":
            break
        if state["status"] in {"failed", "blocked", "interrupted"}:
            save(
                args.output_directory / "status.json",
                dict(status="waiting_for_prior_review", prior=state),
            )
            return
        if time.time() - (prior / "status.json").stat().st_mtime > 600:
            save(
                args.output_directory / "status.json",
                dict(
                    status="waiting_for_prior_review",
                    prior=state,
                    note=(
                        "Prior checkpoint has not advanced for ten minutes; "
                        "inspect before continuing."
                    ),
                ),
            )
            return
        save(
            args.output_directory / "status.json",
            dict(status="waiting_for_prior_run", prior=state, new_retrieval_cases=200),
        )
        import_advanced()
        report()
        time.sleep(15)
    import_advanced()
    for number, q in enumerate(questions, 1):
        for variant in NEW_VARIANTS:
            root = args.output_directory / "variants" / variant / "cases" / q.id
            path = root / "result.json"
            row = json.loads(path.read_text())
            if row["status"] in {"completed", "failed"}:
                continue
            save(
                args.output_directory / "status.json",
                dict(
                    status="answers_running",
                    variant=variant,
                    question=q.id,
                    position=number,
                    total=100,
                ),
            )
            try:
                retrieval = json.loads((root / "retrieval.json").read_text())["result"]
                contexts = json.loads((root / "context.json").read_text())["contexts"]
                retrieval.update(
                    contexts=contexts,
                    matched_children=retrieval["matched_children"][:5],
                    context_tokens=retriever.budget.count("\n\n".join(b["text"] for b in contexts)),
                )
                data = shared.one_pass_answer(q, retrieval, root)
                row["answer"] = data["answer"]
                row["generation"] = {
                    k: data.get(k)
                    for k in [
                        "generation_seconds",
                        "generation_calls",
                        "prompt_tokens",
                        "output_tokens",
                    ]
                }
                judgment = root / "contextual-assessment.json"
                if judgment.exists():
                    judged = json.loads(judgment.read_text())
                else:
                    judged = assess_answer(
                        OllamaRagasJudge(root / "contextual_calls"),
                        dict(
                            question=q.question,
                            reference=q.reference,
                            required_facts=q.required_facts,
                            expected_abstention=q.expected_abstention,
                            evidence="\n\n".join(dict.fromkeys(e.source_text for e in q.evidence)),
                            additional_retrieved_source_text=[b["text"] for b in contexts],
                            answer=data["answer"]["answer"],
                        ),
                    )
                    save(judgment, judged)
                row["metrics"].update(
                    required_fact_coverage=judged["required_fact_coverage"],
                    clean_answer=float(judged["clean_answer"]),
                    unsupported_extra_claims=float(bool(judged["unsupported_extra_claims"])),
                    abstention_correct=float(
                        data["answer"]["insufficient_evidence"] == q.expected_abstention
                    ),
                )
                if not q.expected_abstention:
                    judge = OllamaRagasJudge(root / "ragas_calls")
                    for metric, texts in [
                        ("faithfulness", [b["text"] for b in contexts]),
                        (
                            "citation_support",
                            [retriever.by_id[c].text for c in data["answer"]["citations"]],
                        ),
                    ]:
                        score = (
                            metric_score(
                                root / (metric + ".json"),
                                COLLECTIONS.Faithfulness(llm=judge),
                                user_input=q.question,
                                response=data["answer"]["answer"],
                                retrieved_contexts=texts,
                            )
                            if texts
                            else {"value": None}
                        )
                        row["metrics"][metric] = score.get("value")
                    row["numeric_checks"] = numeric_checks(q, data["answer"]["answer"])
                    row["unit_checks"] = unit_checks(q, data["answer"]["answer"])
                row["status"] = "completed"
            except urllib.error.HTTPError as exc:
                save(
                    args.output_directory / "status.json",
                    dict(
                        status="blocked",
                        http_status=exc.code,
                        question=q.id,
                        variant=variant,
                        note="Cloud error; stopped without automatic retry.",
                    ),
                )
                report()
                return
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            save(path, row)
            print(f"{number}/100 {variant}: {row['status']}", flush=True)
        report()
    save(
        args.output_directory / "status.json",
        dict(status="completed", new_variant_questions=200, reused_advanced_variant_questions=300),
    )
    report()


if __name__ == "__main__":
    main()
