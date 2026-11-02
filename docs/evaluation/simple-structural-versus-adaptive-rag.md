# Simple structural versus adaptive RAG

This is a **secondary downstream study**. The project's primary evaluation is
document parsing: comparing raw models and the final parsing pipeline, with
source/output visual review. RAG results do not establish parser accuracy.

## Saved comparison

The completed study covers 100 questions across five approaches: 200 new
structural-baseline cases and 300 reused advanced cases. All 500 cases completed;
there were no new scoring execution errors. Earlier unavailable scores remain
unavailable. Fixed-size chunk experiments and the superseded first structural
attempt are preserved separately.

The practical baseline repeats source headings on every body continuation,
keeps tables/charts/images separate, repeats captions and table headers, and
includes related explanations when they fit. Splits occur between complete
rows, series/points, paragraphs or sentences. Oversized indivisible units are
preserved and flagged. All actual corpus chunks fit within 500 tokens.

| Approach | Automated strict rubric passes | Mean citation support | Mean answer-model calls |
| --- | ---: | ---: | ---: |
| Simple structure, dense | 88/100 | 0.985 (88 scored) | 1 |
| Simple structure, hybrid | 85/100 | 0.983 (85 scored) | 1 |
| Adaptive hybrid, one pass | 94/100 | 0.968 (88 scored) | 1 |
| Adaptive parent expansion, one pass | 94/100 | 0.663 (88 scored) | 1 |
| Adaptive parent agent | 93/100 | 0.667 (89 scored) | 1.9 |

These are judge scores, not independently verified accuracy. Citation-support
averages exclude unavailable cases, so their denominators differ. All approaches
use the same questions, saved source material, 1,024-dimensional BGE-large
embeddings, 2,400-token context ceiling, Gemma 4 31B answer model and GPT-OSS 120B
judge. The revised rubric passed 90 synthetic control trials, but subsequent
source review found failures those controls did not cover.

## Assessment and limits

Adaptive hybrid with one-pass answering is the current candidate: it retrieved
facts and multi-source evidence missed by the simple methods, without mandatory
agent planning. Simple dense is a credible baseline, including 30/30 automated
table passes. The full agent did not demonstrate an overall gain over one pass
and uses more model calls. Parent expansion often supplies facts that the cited
child itself does not contain, reducing citation support.

A targeted assistant review examined 17 questions and 68 paired answers, plus
one source chart and five image crops. It confirmed retrieval differences but
also found unsupported motion accepted as correct, inconsistent grades for
near-identical image descriptions, and safe refusals undercredited for being
brief. The review is neither independent human validation nor a full manual
rescore. No corrected global accuracy is inferred from it.

Small ranking differences are inconclusive. For example, the agent's observed
five-point strict-pass lead over simple dense has an exploratory grouped-bootstrap
interval of approximately -1.2 to +13.1 percentage points. The questions have
already been inspected; they are not a new unseen holdout. No production pipeline
was automatically changed based on this comparison.

## Records and visualization

- [Machine-readable completed results](../../evaluation/results/rag-structural-comparison-v2.json).
- [Source-review metadata](../../evaluation/results/rag-comparison-source-review-v1.json).
- [Answer-model call counts](../../evaluation/results/rag-comparison-resource-profile-v1.json).
- [Full methodology, calibration and failure analysis](judge-validation-and-rag-ablation.md).
- [Preservation registry](../../evaluation/preserved-runs.json).

The local completed run is
`outputs/quality_evaluations/rag-structural-comparison-v2/`. It contains
`metrics.csv`, `results.json`, `summary.json`, `paired-differences.json`,
`completion-verification.json`, raw judge/answer responses and the frozen runtime
inventory. Its `comparison-with-metric-meanings.html` shows definitions in the
rightmost column; `paired-source-review-v1/report.html` shows the paired review.

Checksum-inventoried copies of the completed run, failed/superseded attempts and
source review are preserved under `outputs/evaluation_archive/gpt-oss-120b_output/`.
Private PDFs, question/evidence records and raw generated outputs stay local,
outside Git and package distributions. This is local preservation, not a remote
backup or a claim that the repository has been published.
