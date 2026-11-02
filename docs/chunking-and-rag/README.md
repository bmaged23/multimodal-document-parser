# Chunking, retrieval and RAG

[Project overview](../../README.md) · [Parsing evaluation](../parsing-evaluation/README.md) ·
[Parsing pipeline](../parsing-pipeline/README.md)

This is a **secondary downstream study** using saved parser outputs. The project's
primary focus remains parsing-model comparison, pipeline evaluation and source/output
visualization. RAG results do not establish OCR accuracy.

## Completed comparison

The study evaluates 100 questions over five approaches: 200 new structural cases
and 300 reused advanced cases. All 500 completed, with no new scoring execution
errors. Earlier metric missingness is retained rather than zero-filled.

| Approach | Strict automated rubric passes | Citation-support mean | Answer-model calls/question |
| --- | ---: | ---: | ---: |
| Simple structure + dense | 88/100 | 0.985 (88 scored) | 1 |
| Simple structure + hybrid | 85/100 | 0.983 (85 scored) | 1 |
| Adaptive hybrid, one pass | 94/100 | 0.968 (88 scored) | 1 |
| Adaptive parent expansion, one pass | 94/100 | 0.663 (88 scored) | 1 |
| Adaptive parent agent | 93/100 | 0.667 (89 scored) | 1.9 |

The current candidate is adaptive hybrid with one-pass answering. The full agent
did not demonstrate an overall improvement and makes more model calls. These are
exploratory judge results, not independently verified accuracy or a universal
winner. Citation averages have different denominators. The
[saved comparison](../evaluation/simple-structural-versus-adaptive-rag.md) and
[full methodology/audit](../evaluation/judge-validation-and-rag-ablation.md)
explain these qualifications.

## Inputs and evidence

The index uses 48 saved pages: the original two-page multimodal sample, 20 IRENA
pages and 26 IMF appendix pages. The canonical parsed documents and source-page
mappings are retained. This study did not re-run parsing to create each RAG variant.

The question mix is 20 narrative, 30 table, 15 chart, 5 image, 20 cross-page and
10 unanswerable questions. All 100 are frozen evaluation questions; separate
synthetic examples check evaluator operation. Source-derived silver references,
expected evidence groups, table cells and chart points are saved locally. GPT-OSS
reference drafts are separate from the scoring targets. These questions have
already been inspected and cannot become a fresh unseen holdout for tuning.

## Chunking approaches

| Property | Practical simple structure v2 | Adaptive parent/child |
| --- | --- | --- |
| Text | Section heading hierarchy repeats on every continuation | Searchable children retain evidence offsets and contextual parts |
| Tables | Separate physical tables; repeated captions/column headers | Continued tables can share logical parents; column groups stay separate |
| Charts/images | Separate groups, repeated headings/captions | Visual parts and exact series/point evidence retained |
| Related explanations | Unique explicit/layout links; whole notes included when space permits | Context parts and deferred evidence retained according to production policy |
| Splitting | Rows; series or complete points; paragraphs/sentences | Production source slices preserve characters; behavior differs from experimental v2 |
| Oversized units | Preserve intact and flag a soft-limit exception | Follow production limits and warnings; not silently replaced by v2 |
| Retrieval expansion | None: retrieve preassembled standalone chunks | Optional bounded parent expansion |

The simple baseline targets 350 tokens and a soft maximum of 500, including
headings. Optional repeated explanation context is capped at 100 tokens and
cannot displace primary content. Longer explanations remain searchable under the
related item's scope. Ambiguous associations remain separate. No actual corpus
chunk needed an oversized exception. Its three bundles contain 12, 79 and 321
chunks. Inferred table-header handling and explanation links are disclosed
representation differences, not a pure chunk-boundary-only ablation.

The saved adaptive index contains 88 parents, 692 children and 693 embedding
windows. Earlier fixed-size and superseded structural attempts remain diagnostic
baselines; they are not the main comparison with a reasonable simple system.

## Embedding and retrieval

All compared variants use **BAAI/bge-large-en-v1.5**, exactly **1,024 dimensions**,
L2-normalized CLS vectors and verified CUDA execution. The ONNX export revision
is `e93b9305e013fdf771b471426516285da0cca188`. BGE's WordPiece limit is separate
from the chunk tokenizer: lossless embedding windows contain at most 480 BGE
tokens, with offsets to the original child. Retrieved context reconstructs the
child rather than silently discarding its tail.

Dense retrieval scores vectors; keyword retrieval uses SQLite FTS5 BM25. Hybrid
uses equal-weight reciprocal-rank fusion with k=60, not a learned reranker.
Answer generation uses top five hits and a 2,400-token formatted context ceiling.
Top-10 diagnostics and actual bounded-context source coverage are saved separately.
The simple hybrid configuration sometimes demotes useful dense hits, so hybrid
is not assumed superior without case-specific evidence.

## Answering and the agent

The answer model is **Gemma 4 31B on Ollama Cloud**. One-pass baselines use the
same generation prompt, temperature zero and 1,600-token output ceiling.
The agent can answer, rewrite once or abstain, with at most two retrieval calls.
It checks citation IDs against supplied evidence; that checks identifiers, not
whether the cited child supports the claim. Parent expansion can make this
distinction especially important.

The agent is a separate service from ingestion. It does not upload or parse PDFs
inside a question request. [API architecture](../architecture/service-pipeline.md)
documents artifact IDs, authentication, durable jobs and storage boundaries.

## Evaluation and judge validation

| Metric source | Measures |
| --- | --- |
| Deterministic checks | Expected source coverage, retrieval completeness, citation IDs, numeric/unit mentions and abstention flags |
| Ragas + GPT-OSS 120B | Faithfulness; citation support by applying faithfulness to cited chunks |
| Original Ragas diagnostics | Answer relevancy, context precision/recall and factual correctness; the original correctness setup failed calibration and is excluded from ranking |
| Custom contextual rubric + GPT-OSS 120B | Required-fact coverage, unsupported extra claims and answer/refusal decisions |

The revised custom rubric passed 90 trials across 30 controls, including held-out
synthetic cases. That finite check did not certify all corpus judgments. A
17-question/68-answer assistant review, including one chart and five photographs,
found unsupported motion accepted as correct, inconsistent image grades and safe
refusals penalized for missing reference explanations. No corrected global
accuracy is inferred from that selected review. Group-bootstrap intervals and
reference/judge limitations accompany the saved scores.

## Artifacts, visualization and reproduction

- [Completed metrics](../../evaluation/results/rag-structural-comparison-v2.json).
- [Call-count evidence](../../evaluation/results/rag-comparison-resource-profile-v1.json).
- [Source-review metadata](../../evaluation/results/rag-comparison-source-review-v1.json).
- [Development index/service checks](../evaluation/adaptive-rag-development.md).
- [Full reproduction commands](../evaluation/judge-validation-and-rag-ablation.md#reproduction).

The local completed run is
`outputs/quality_evaluations/rag-structural-comparison-v2/`. Open
`comparison-with-metric-meanings.html` through a local HTTP server for filtering
and rightmost metric explanations. `paired-source-review-v1/report.html` contains
paired answers and review notes. CSV, JSON, raw judge/answer responses, expected
evidence, configuration and a frozen runtime inventory are retained. Full private
artifacts are archived by model under the
[preservation registry](../../evaluation/preserved-runs.json).

Use the existing uv environment with `rag` and `evaluation` extras; cloud scoring
requires the configured credential, and embedding requires CUDA. Use fresh run
directories when configurations change. No additional parsing, model rerun or
production change is required just to inspect these completed comparisons.
