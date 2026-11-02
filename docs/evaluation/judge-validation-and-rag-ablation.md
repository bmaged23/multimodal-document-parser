# Judge validation and simpler RAG alternatives

The original 100-question run is a preserved baseline, not a tuning target whose
scores can be assumed correct. A subsequent 20-case assistant audit found both
evaluation defects and actual pipeline failures. No OCR or visual enrichment
was rerun for this work.

## Audit findings

The audit deliberately samples disagreement cases, tables, charts, photographs,
cross-page questions and refusals. It is not a random sample, an independent human
review or a new visual adjudication of the source PDFs.

- In `q001`, the answer correctly lists the two requested risks. During Ragas
  reference decomposition, the judge invents a relationship between manufacturing
  dynamics and tariffs. The resulting zero correctness score is misleading.
- In `q021` and `q029`, the candidate supplies entity/technology scope present in
  the question but omitted from the short reference. Judging the reference in
  isolation produces misleading mismatches despite matching source values.
- In `q012`, `q057` and `q062`, the answer uses information present in expanded
  context but cites a child that does not contain the relevant rule or series.
  Valid citation identifiers do not establish citation support.
- In `q068`, a generated image description asserts a direction of motion that a
  still image does not establish. An answer can faithfully repeat this claim
  while remaining unsupported by the actual visual evidence.
- In `q082`, the final context lacks one required caveat. A correct number in the
  answer is also accompanied by a citation to the wrong table column group.
- In `q090`, a valid percentage is calculated from the source amounts. A judge
  must allow supported arithmetic rather than require a verbatim printed result.

Private artifacts: `outputs/quality_evaluations/ragas-100-v1/audit-20-v1/`.

## Evaluator calibration, before comparing answer quality

The judge is GPT-OSS 120B through Ollama Cloud. Requests, raw responses, controls,
expectations and thresholds are saved without API credentials. Repeats use
separate request-cache directories so they are actual independent API calls,
not repeated reads of one cached response. Cloud model revisions are not pinned.

| Evaluation setup | Result | Interpretation |
| --- | --- | --- |
| Original Ragas FactualCorrectness | 13/30 explicit known-answer checks passed in the initial validation | Do not use this configuration for tuning or winner selection. Correct short-reference answers sometimes score zero. |
| Contextual rubric v1 | Fact coverage and unsupported-claim checks passed 60/60; answer/refusal decision failed 48/60 | Failed calibration. The boolean field name was ambiguous; preserved rather than silently replaced. |
| Contextual rubric v2 | 90/90 control trials passed; component decisions consistent for all 30 controls | Cleared the preregistered control gate for an exploratory comparison, not universal judge certification. |

The first 20 controls cover exact answers, paraphrases, missing facts, incorrect
numbers, swapped associations, wrong units, extra supported/unsupported detail,
unsupported motion and refusals. Version 2 clarifies the answerability decision
in the output schema and adds 10 new held-out controls, including different
entities, arithmetic and an instruction-injection control. Every control is
tested three times. Frozen expectations and tolerances are saved before calls.
The controls are assistant-authored, not independently human-labelled. The
held-out controls assess this finite rubric; they do not establish general
accuracy over arbitrary documents or visual content.

The revised rubric is a **custom metric**, separate from Ragas. It interprets
reference phrases in question context, evaluates required facts without
decomposing references in isolation, and assesses unsupported extra claims and
the answer-or-refuse decision separately. It preserves equivalent number formats
and checks associations, not just the presence of numbers. Required reference
facts have mixed granularity; coverage is over those saved facts, not guaranteed
uniform atomic claims.

Private calibration artifacts:

- `outputs/quality_evaluations/judge-validation-v1/`
- `outputs/quality_evaluations/judge-validation-v2/`

The original Ragas scores remain intact. Existing Ragas faithfulness and
citation-support scores are reported separately; calibration of factual
correctness does not certify those other judge tasks. All text-judge results
remain distinct from independent verification of PDFs and images.

## Controlled five-way comparison

| Variant | Chunks | Retrieval | Answering |
| --- | --- | --- | --- |
| `fixed-dense-plain` | 350 tokens, 50-token overlap | Dense | One generation |
| `fixed-hybrid-plain` | Same fixed chunks | Dense + BM25 | Same one-pass prompt |
| `adaptive-hybrid-plain` | Existing adaptive children | Dense + BM25, no whole-parent expansion | Same one-pass prompt |
| `adaptive-parent-plain` | Existing adaptive children | Dense + BM25, parent expansion | Same one-pass prompt |
| `adaptive-parent-agent` | Existing adaptive children | Production retrieval and bounded rewrite | Reuse the saved production-agent answers |

All variants use the same 100 frozen questions, saved source representations,
BGE-large-en-v1.5 embeddings with 1,024 dimensions, top-five answer retrieval,
2,400-token context ceiling, Gemma 4 31B and GPT-OSS 120B judge. Embedding model
revision and index checksums are recorded. Generation uses temperature zero and
the same 1,600-token output ceiling. Variants are interleaved by question to
reduce time-of-run confounding; the reused production baseline necessarily
comes from an earlier run. Only one answer per method/question is generated.

The fixed baseline flattens the **same saved evidence-part text**, including
table-row labels and chart descriptions, in original parent/part order. This
holds parsing, enrichment and serialization constant. It is not a claim about
raw Markdown chunking, a different parser, all fixed chunk sizes or every simpler
RAG implementation. The no-expansion adaptive baseline retains its existing
header/caption context, which is part of that chunk representation.

Expected evidence is matched through document/element coordinates, table cells,
chart series and text spans, not by comparing incompatible chunk identifiers.
The original conservative full-part metric is retained as a diagnostic. A
second cell-aware measure credits complete requested column-label/value segments
even when unrelated columns in a long row are outside a fixed window. Fragments
can be joined across retrieved chunks; missing gaps never receive full credit.
Narrative and chart evidence still require complete saved parts. Scores are
therefore conservative source-inclusion checks rather than perfect relevance
annotations. Actual bounded answer contexts are reported separately from
top-k retrieval before context packing.

The initial bounded-context retrieval comparison, over 90 answerable questions:

| Variant | Complete expected evidence present |
| --- | ---: |
| Fixed dense, plain RAG | 60.0% |
| Fixed hybrid, plain RAG | 66.7% |
| Adaptive hybrid, plain RAG | 93.3% |
| Adaptive parent, plain RAG | 94.4% |
| Adaptive parent, production agent | 95.6% |

## Practical structural baseline added after review

Fixed-size windows are retained as diagnostic lower baselines, not the main
representation of a reasonable simple RAG system. Following review of the initial
comparison design, `simple-structure-v1` adds two stronger one-pass baselines:
`simple-structure-dense` and `simple-structure-hybrid`.

- Text chunks keep the full source heading hierarchy. When a section is split,
  every continuation repeats exactly the same hierarchy until the source section
  changes. Greedy paragraph packing targets 350 tokens with a 500-token ceiling,
  including repeated headings. Individually oversized paragraphs are split
  without losing source characters.
- Each physical table is a separate group. Oversized tables split at row
  boundaries where possible, with the caption and column headers repeated in
  each chunk. Table footnotes remain separately searchable under that table.
- Each chart/image remains a separate group with its caption and heading context.
  Oversized visual descriptions split within that group, retaining their source
  offsets and generated/extracted distinctions.
- Retrieval returns the standalone chunks as indexed. There is no parent
  expansion, logical-table joining across pages, query rewrite or agent planning.

The same saved canonical documents are reused; no parsing or enrichment is
rerun. The three document bundles contain 13, 87 and 321 chunks respectively.
Validation checks source coverage, token bounds and repeated continuation context.

The structural baseline also treats a blank stub in an otherwise annotated table
header band as inferred header structure. This avoids degrading real column names
to generic positions because of an empty corner cell. It operates on a copy;
source annotations and the existing production chunker remain unchanged. This,
and moving table notes out of every row group, are disclosed representation
differences: the new comparison is a practical-baseline comparison, not a pure
boundary-only ablation.

The new run evaluates 100 questions for each structural retrieval variant and
reuses all three saved adaptive/parent/agent answer sets and scores. It waits for
the prior cloud batch before generating new answers. The old fixed-window runs
remain preserved separately. The same generation prompt, model, context ceiling,
embedding revision and validated custom judge are used. Configuration and runtime
code are frozen before the new index and cloud evaluation. These baseline choices
were added after initial results were inspected, so the comparison remains
exploratory rather than an unseen holdout test.

Artifacts: `outputs/quality_evaluations/rag-structural-comparison-v1/`.
Its live report is `comparison-with-metric-meanings.html`; continuation-context
and source-coverage results are stored in each bundle's `validation.json`.

### Revised semantic boundaries and related explanations (v2)

The initial five-way comparison completed all 500 variant/question cases. The
first practical structural run was then stopped and marked superseded after the
user requested related explanatory text and stricter semantic split boundaries.
Its 18 completed new-method cases and all partial artifacts remain preserved;
they are not mixed into the replacement run. The 300 reused advanced cases in
that directory are distinct from those 18 new-method cases.

`rag-structural-comparison-v2` freezes the revised policy:

- Keep an entire table or visual together when it fits the ceiling. Otherwise
  split tables between complete rows and charts between complete series where
  possible, or complete point records when a series is too large. Never split
  the bytes of a cell, point record, formula or JSON analysis block.
- Split oversized body paragraphs/descriptions at heuristic paragraph/sentence
  boundaries. Preserve an individually oversized sentence or row intact and flag
  it explicitly rather than cut it to satisfy a token counter. The 500-token
  ceiling is soft only for those audited exceptions. Production validation stays
  strict by default; the experimental caller explicitly supplies exception IDs.
- Carry the source heading hierarchy, captions and table column headers into
  each continuation. Adjacent source captions that were separate document
  elements are linked conservatively and repeated, not invented by an LLM.
- Use explicit table-footnote links first. Otherwise associate a paragraph only
  with a unique same-page/section caption reference, or immediately adjacent
  below-text with strong column overlap. Body text/headings block proximity
  associations, and side-by-side visuals remain competing candidates. Mark
  inferred associations and leave ambiguous text standalone.
- Uniquely associated explanations remain searchable under their related item's
  scope. Repeat a whole explanatory block inline only when it fits without
  displacing main content, using at most 100 tokens for repeated explanation
  context. Longer explanations get their own semantic chunks with the item's
  headings/caption. Store source-element and child links in
  `related_explanations.json`.

The saved corpus produces 12, 79 and 321 chunks, with maximum sizes 256, 499 and
500 tokens respectively. No actual corpus chunk needed an oversized exception.
The relationship audit records 11 adjacent source captions, 78 explicit table
footnote associations, 9 explicit caption references and 10 nearby below-text
associations. These rule-based links are not independent semantic annotations;
the source spans and association methods are preserved for review.

The final runtime is frozen before re-indexing/scoring. The new dense/hybrid runs
evaluate all 100 questions each and reuse the already completed adaptive results.
No parsing, visual enrichment or advanced answer generation is rerun. The live
report and all metrics are under
`outputs/quality_evaluations/rag-structural-comparison-v2/`.

### Completed v2 comparison

All 500 variant/question cases completed: 200 new structural-baseline cases and
300 reused advanced cases. There were no case failures or new scoring errors.
Previously unavailable reused metric values remain unavailable, including the
original agent's faithfulness error; they are not imputed as zero.

| Approach | Custom clean-answer rate | Mean required-fact coverage | Ragas citation support (scored cases) |
| --- | ---: | ---: | ---: |
| Simple structural dense | 88/100 | 0.8933 | 0.9855 (88) |
| Simple structural hybrid | 85/100 | 0.8550 | 0.9826 (85) |
| Adaptive hybrid, one pass | 94/100 | 0.9450 | 0.9678 (88) |
| Adaptive with parent expansion, one pass | 94/100 | 0.9450 | 0.6631 (88) |
| Adaptive parent agent | 93/100 | 0.9350 | 0.6666 (89) |

Clean-answer rate requires all saved required facts, no contradicted required
facts or unsupported extra claims, and an appropriate answer/refusal decision.
It is a custom text-judge result, not independent human accuracy. Required-fact
coverage includes the saved refusal references and uses mixed-granularity facts.
Citation-support averages condition on available scores; refusals without
citations and other unavailable cases are excluded. The denominators differ,
so those conditional citation means must not be read as overall answer success.

The observed results do not show an advantage for agentic planning over the
adaptive one-pass baseline. They also retain the earlier citation-granularity
problem with parent expansion. Do not generalize a winner from these 100 inspected
questions: the agent-minus-structural-hybrid clean-answer difference is +0.08,
with an exploratory group-bootstrap interval of approximately -0.044 to +0.171.
That interval includes zero. Extra model calls, source-format differences and
run-date drift remain relevant tradeoffs.

Public summary: `evaluation/results/rag-structural-comparison-v2.json`.
Private exports include `metrics.csv`, `results.json`, `summary.json`,
`paired-differences.json`, `completion-verification.json` and the live HTML report
with a rightmost metric-meaning column. No production pipeline was automatically
changed after observing these results.

### Post-run paired source review

A subsequent assistant review examined 17 selected questions and 68 paired
answers, using saved source text/cells and direct inspection of one chart and
five photograph crops. This was not a random sample, an independent human review
or a full-corpus rescore. Original scores remain unchanged.

The review confirms genuine retrieval differences: simple methods miss India's
2011/12 base-year fact and the global USD 467 billion denominator needed for
China's avoided-cost share. Simple dense nevertheless passes all 30 table
questions under the rubric, making it a credible baseline. Rank traces for
`q035` show keyword fusion demoting the correct dense top result, explaining one
simple-hybrid failure without implying hybrid retrieval is universally worse.

Remaining judge defects were also found despite passing finite synthetic controls:

- `q068`: all methods assert a component is being lowered. A still image does
  not establish that motion, yet all receive a pass.
- `q070`: near-identical dam descriptions get different binary outcomes.
- `q067`: reasonable scene descriptions receive zero for omitting a reference
  caveat about floating that the question did not explicitly request.
- `q094`, `q099`, `q100`: safe generic refusals are undercredited for omitting
  reference explanations. A binary zero is not evidence of hallucination.

Adaptive hybrid with one-pass answering is the current practical candidate,
balancing observed completeness, child-citation support and one answer-model call.
The agent averages 1.9 calls. It helps individual cases, but an overall advantage
has not been demonstrated; one-point score differences are especially unreliable.
This recommendation does not change production or trigger another parsing run.

Private audit: `paired-source-review-v1/` inside the v2 comparison directory.
Public metadata: `evaluation/results/rag-comparison-source-review-v1.json`.
The live report flags these findings; the earlier frozen result archive is intact.

The earlier retrieval figures establish source availability under the coverage
definition, not answer correctness or overall superiority. Answer quality is
reported separately above. Missing/error scores are excluded rather
than zero-filled; completion counts must accompany every mean. Paired
differences use a fixed-seed bootstrap over source-related question groups,
with 2,000 resamples, and are exploratory without multiple-comparison correction.
Future tuning would require fresh verification questions: these 100 have already
been inspected and cannot serve as a new unseen holdout.

Private comparison artifacts:
`outputs/quality_evaluations/rag-approach-comparison-v1/`.
The runtime snapshot and SHA-256 inventory preserve the execution code before
comparative answer generation. Raw model artifacts and private documents are
excluded from Git and package distributions.

## Reproduction

Use the existing uv environment with the RAG and evaluation extras installed.
Cloud calls require the configured Ollama credentials; embedding requires CUDA.
The source runs, adaptive bundles and index must already exist locally.

```bash
uv run --extra rag --extra evaluation python scripts/audit_rag_evaluation.py
uv run --extra rag --extra evaluation python scripts/validate_rag_judge.py \
  --rubric-version v2 --repeats 3 \
  --output-directory outputs/quality_evaluations/judge-validation-new
uv run --extra rag --extra evaluation python scripts/compare_rag_approaches.py \
  --judge-validation outputs/quality_evaluations/judge-validation-new/summary.json \
  --output-directory outputs/quality_evaluations/rag-approach-comparison-new
uv run --extra rag --extra evaluation python scripts/report_rag_retrieval_comparison.py \
  --directory outputs/quality_evaluations/rag-approach-comparison-new
uv run --extra rag --extra evaluation python scripts/compare_structural_rag_baseline.py \
  --prior-comparison outputs/quality_evaluations/rag-approach-comparison-new \
  --output-directory outputs/quality_evaluations/rag-structural-comparison-new
```

Use fresh output directories for changed controls, configurations or explicit
retries. Retain failed attempts. The existing run can be resumed using its
frozen runtime snapshot rather than modified development code.
