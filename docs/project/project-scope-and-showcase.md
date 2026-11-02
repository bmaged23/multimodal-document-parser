# Project scope and evaluation showcase

## Primary purpose

Evaluate and compare multimodal document parsing models, then evaluate the final
parsing pipeline. A visitor should quickly see model configurations, measured
results, source/output examples and where the system fails.

Keep two evaluation layers distinct:

1. **Raw parsing models:** text, reading order, tables/spans/headers, visual-region
   detection, completeness/failures, per-page timing and sampled GPU memory.
2. **Final parsing pipeline:** normalization, visual descriptions/chart values,
   source correlation, exports and recovery. Mark inferred and corrected content;
   never present repaired pipeline scores as raw model accuracy.

## Presentation order

1. A real source/output review preview showing a complex table or chart.
2. A concise measured parser-comparison table with case size and configuration.
3. Selected quality checks and a visible before/after pipeline example.
4. Failure cases and measurement limitations beside the results.
5. A compact secondary chunking/retrieval/agent comparison immediately after
   parsing results, with links to the complete supporting study.
6. Reproduction instructions, architecture, CI and deployment status.

The detailed reading paths are [parsing evaluation](../parsing-evaluation/README.md),
[parsing pipeline](../parsing-pipeline/README.md) and
[chunking/RAG](../chunking-and-rag/README.md), indexed by the
[documentation README](../README.md).

Use saved evaluation artifacts and source-attributed public examples. The README
screenshots are actual review renders from the source-reviewed IRENA selection;
they are not fabricated model predictions. The chart view identifies its native
PDF corrections and the table view identifies inferred header roles. Keep private
documents and full generated artifacts outside Git.

The standalone screenshots are under `docs/assets/`; their source and capture
details are recorded in `docs/assets/readme-gallery.provenance.json`.
The main README shows one table preview and keeps four varied examples collapsed.
The public saved-output demo is linked immediately after the introduction. The comparison
summary is `evaluation/results/parser-comparison-showcase-v1.json`.

## Secondary scope

Chunking, embeddings, retrieval and the RAG agent demonstrate downstream use of
the parsed output. Preserve the completed comparison and evidence, but do not
let RAG feature expansion displace parsing evaluation or the main showcase.
The [saved RAG comparison](../evaluation/simple-structural-versus-adaptive-rag.md)
records both the measured results and the subsequent source-review caveats.

## Next delivery focus

Continue the parsing review/evaluation and release milestones: make existing
comparisons easy to inspect, consolidate case/model coverage and failures, and
document reproduction and deployment evidence. Prefer existing saved runs over
expensive re-parsing. Expand reference annotations or rerun a model only when a
specific evaluation gap warrants it. Docker deployment, hosted demonstrations
and independent human benchmarks must be marked unverified until actually run.
