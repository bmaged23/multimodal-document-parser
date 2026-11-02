# Retired experiment: one-page unquantized dots.mocr CPU probe

This approach has been retired at the user's request. Active CPU code, launch
profile, tests, and downloaded weights were removed; frozen diagnostics and
reproduction snapshots remain. GPU quantized inference is the supported path.

The approved CPU-only experiment ran on source page 1 after WSL's memory cap
was raised to 12 GiB (11.68 GiB observed). **Model execution fit in memory, but
extraction failed.** No completed normalized page or full-page accuracy score
is claimed.

## Controlled inputs and storage verification

Both language weights and the vision projector came from
`prithivMLmods/dots.mocr-GGUF` revision
`5cd9f01392968881af86dfde7bc60b9cb412f87a`, matching the conversion revision used
by the quantized baseline. SHA-256 and file sizes were checked. GGUF tensor
headers were independently inspected: language tensors were 141 F32 + 198 BF16;
projector tensors were 92 F32 + 212 BF16. Neither file contained quantized tensor
types. Storage verification does not establish runtime arithmetic precision.

The native runtime was llama.cpp b11960 built with CUDA 12.6/sm75, executing its
CPU backend. CUDA devices were hidden, GPU layers set to zero, and projector
and operation offloading disabled. Six CPU threads, 8,192 context tokens, no
flash attention, 4,096 output-token cap, and a 15-minute inference deadline were
used. This is an unquantized **GGUF CPU configuration**, not a verified original
Transformers execution. The converter does not pin its original checkpoint revision.

The selected PDF has its own document hash and an explicit mapping to original
source page 1. RGB source pixels matched the original preparation exactly.
The actual sent 840 × 1,176 PNG and official prompt hashes also matched the saved
quantized run. The quantized reference reuses its existing page-1 prediction;
its timing and full-run GPU-memory scope were retained without new inference.

## Outcome

| Observation | Result |
| --- | --- |
| CPU inference attempt | 728.46 seconds, about 12 min 8 s |
| Generated tokens | 4,096; finish reason `length` |
| Peak server RSS / kernel high-water observation | 7.586 GiB |
| Model-process swap | 0 KiB |
| Memory guard | Not triggered |
| Completed normalized pages | 0 / 1 |
| Quantized GPU reference | Completed page 1 in 33.77 s |

RSS includes mapped model-file pages, scratch buffers, and startup. GPU and CPU
memory measurements cover different resources. The failed CPU attempt generated
more tokens than the successful GPU reference, so their timings are not a clean
throughput comparison.

The CPU response recognized initial text and emitted a table, then repeated
chart-caption text instead of finishing the page. Inspection found 78 occurrences
of `Peak: CZ-29`. Among 86 complete object prefixes in the truncated response,
all 86 omitted the required `category`; 66 extended below the 1,176-pixel image.
These counts describe raw-response diagnostics, not accepted parsed elements.
No structured table-check or CER/WER score was computed from the invalid output.

Increasing the output cap would likely extend this observed loop. This run was
therefore stopped at its configured limit and preserved as a failure. No automatic
retry or silent JSON repair was performed. The server was terminated afterward.

Removing quantization did not improve this configuration into a usable parser.
This does not prove the original model is inherently worse or establish that
quantization alone caused earlier GPU boxes: CPU/GPU arithmetic, conversion,
and backend implementation remain potential factors. Original Transformers
inference would need a separate controlled validation.

## Preserved diagnostics

Working run: `outputs/parser_runs/dots-mocr-bf16-cpu-page1-01/`.
`diagnostic-review.html` shows the original PDF at left and the escaped incomplete
response at right. It is explicitly a failed-run review, not a parsed-document
report. Full response JSON, sent PNG, prompt/runtime metadata, CPU resource
measurements, weight-type verification, qualitative findings, and server logs
are retained.

Frozen snapshots are under
`outputs/evaluation_archive/dots.mocr-BF16-mmproj-BF16-CPU_output/`.
The matched reused quantized reference is separately archived in its existing
Q4/Q8 namespace. Original baselines and all prior snapshots remain unchanged.
