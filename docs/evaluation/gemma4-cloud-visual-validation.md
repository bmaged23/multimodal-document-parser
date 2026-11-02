# Gemma 4 31B cloud visual validation

The default visual model is `gemma4:31b` on Ollama Cloud, a 31B-class multimodal
model. The original two-page private synthetic PDF's three visual regions were
sent with marked page context using the user's authorized Ollama credential.
This is a development spot check, not a held-out benchmark.

Run: `outputs/parser_runs/paddleocr-gemma4-31b-visual-enrichment-01/`.
All three generations finished and passed schema/semantic validation. The 9/9
selected source-derived assertions cover three visual types, three printed chart
values (18.4, 23.1, 31.7 ms at 35%, 55%, 75% humidity), and three flowchart edges
including the YES/NO branches. These assistant source-image checks are not
independent human ground truth and were not supplied as model input.

Manual review confirmed one correctly grouped latency series, the sensor/error/
review/accept flowchart nodes and connections, and a synthetic micrograph description
of outlined circles plus its printed 50 um scale bar. The description avoids
identifying a real organism. The chart has no printed y-axis title; the model left
`y_axis=null` while retaining the latency unit from the chart explanation.

This is a better usable result than the earlier local experiments on these three
visuals. Routing and prompt/schema changes also occurred, so this does not isolate
a model-only quality improvement. It does not establish general OCR, handwriting,
complex chart, or image-description accuracy. NASA and diary enrichment are not
part of this cloud smoke test. The original Paddle OCR output is reused, not rerun.

Each image input and hash, source region, prompt and response schema is retained.
Native cloud responses preserve the returned model alias and provider metadata.
The catalog digest and alias are recorded, but do not pin hosted deployment weights.
Do not interpret cloud timing fields as measured laptop GPU performance.
No local model weights were downloaded and no local model-unload operation is sent.

Ollama Cloud currently does not enforce structured outputs. JSON is requested in
the prompt and validated locally. A single complete Markdown JSON fence may be
unwrapped; the original response is retained and the normalization is explicitly
recorded. Partial JSON, extra prose, wrong graph references, invalid values and
forbidden estimates are not repaired. Failures stay visible in the enrichment
manifest. Exact effective prompts and schemas enable inspection of this behavior.

The combined model namespace is
`outputs/evaluation_archive/PaddleOCR-VL-1.6+gemma4:31b_output/`.
Later documentation/review notes create a new immutable snapshot; the initial
snapshot and prior local-model attempts remain untouched. Credentials stay in
the ignored `.env`; archive copying excludes credentials and local model caches.

Official sources:
[Gemma model listing](https://ollama.com/library/gemma4),
[cloud authentication](https://docs.ollama.com/cloud),
[structured-output limitation](https://docs.ollama.com/capabilities/structured-outputs).
