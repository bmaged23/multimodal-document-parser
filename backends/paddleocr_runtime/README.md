# PaddleOCR inference runtime

This uv project isolates GPU/framework dependencies from the document application.
It runs the **complete** PaddleOCR-VL-1.6 pipeline (layout detection plus VLM),
one rendered input page at a time. The application launches it as a subprocess.

From the repository root, on compatible Linux/WSL NVIDIA hardware:

```bash
uv sync --project backends/paddleocr_runtime --locked
uv run document-parser parse outputs/parsed_documents/<document-hash>-dpi-144
```

The lockfile pins PaddleOCR 3.6.0, PaddleX, PaddlePaddle GPU 3.2.1, and CUDA
runtime dependencies. The GPU package is resolved from Paddle's explicit CUDA
12.6 index. Downloads are large; initial execution also downloads model weights.
Downloaded models live under the repository's ignored `.cache/` directory.
Do not install a CPU Paddle package in this environment alongside the GPU package.

`--device cpu` selects CPU execution using the installed runtime, but CPU-only
installation and performance have not been validated. GPU support depends on
architecture, driver, precision, and available memory, not only Python version.

`run_inference.py` accepts a prepared bundle and a new raw-result directory.
It saves `page_0001.json`, subsequent pages, and `inference-manifest.json`.
The manifest separates initialization from per-page time. GPU memory is total
GPU 0 usage sampled once per second, including other processes; it is not an
allocator peak. A first run includes model downloads in initialization time.

Document orientation classification and unwarping are disabled because normalized
source regions must align with the already-rendered preparation images. Native
PDF rotation has already been applied during preparation. The adapter rejects
raw dimensions that do not match those images.

The main application imports no Paddle libraries. CI exercises saved synthetic
results; actual GPU evaluations run separately. No model accuracy score follows
from passing these integration tests.

Official setup and output reference:
https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html
