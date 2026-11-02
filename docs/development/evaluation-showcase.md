# Browse saved parsing evidence

Build a portable static viewer from preserved results, without rerunning models:

```bash
uv run --no-sync python scripts/build_evaluation_showcase.py \
  --output-directory outputs/evaluation_showcase/my-showcase
uv run --no-sync python -m http.server 8765 --bind 127.0.0.1 --directory outputs
```

Open `http://localhost:8765/evaluation_showcase/my-showcase/index.html`.
Reuse an existing server on port 8765 rather than starting a second one.
The build requires the local saved runs listed in `evaluation/showcase-cases.json`;
they are excluded from Git and the container. A fresh checkout alone does not
contain those predictions. Missing runs remain explicitly unavailable.

Use **Expand review** to give the PDF and extraction the window. Drag the divider
to resize them; zoom the source and output independently. Comparison uses model
or stage tabs by default, with optional side-by-side output. Page navigation
synchronizes the source and extraction and retains original publication pages.
Content, Markdown preview/source, canonical JSON and saved raw replies are separate.
Source-region buttons highlight the corresponding PDF area.

The default public-source selection includes NASA, a handwritten NPS diary,
IRENA and IMF. Private examples require explicit `--include-private` and must stay
local. Bundles contain source PDFs and outputs: review source permissions before
hosting them. External images from untrusted model Markdown are blocked; the
viewer has no remote fonts or JavaScript dependencies.

PaddleOCR, quantized dots.mocr and experimental Unlimited-OCR output are labeled
separately. dots.mocr labels disclose Q4_K_M (4-bit) language weights and
a Q8_0 (8-bit) vision projector. Final source-reviewed pipeline stages include repairs and must not be
presented as raw-model accuracy. Accepted-page counts are not quality scores.
The RAG comparison is a secondary tab with saved metrics and judge limitations.

`build-manifest.json` records copied-file hashes, private-input inclusion and zero
inference requests. `catalog.json` preserves sources, stages and measurement scopes.
Browser tests cover model switching, page mapping, failed output, untrusted text,
expanded geometry, resizing, zoom, comparison tabs and mobile overflow.
