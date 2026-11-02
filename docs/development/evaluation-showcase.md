# Browse saved parsing evidence

**[Open the public demo](https://bmaged23.github.io/multimodal-document-parser/)**
or [download the portable ZIP](https://github.com/bmaged23/multimodal-document-parser/releases/download/v0.1.0-rc.2/public-demo-v1.zip).
It contains seven selected public-source pages, all three OCR configurations,
and IRENA OCR/enriched/source-reviewed stages. It runs no live model requests.

## Public demo versus the full local explorer

The public bundle is a curated excerpt: original page mappings, failures and
source attribution remain attached. Completion/timing cards describe the full
saved evaluation selection, not just the displayed excerpts. Downloadable JSON
is explicitly per-page excerpt data rather than a complete canonical Document.
The complete local explorer below needs the ignored saved runs.

## Build the full local explorer

Build a portable static viewer from preserved results, without rerunning models:

```bash
uv run --no-sync python scripts/visualization/build_evaluation_showcase.py \
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

## Rebuild the curated public bundle

```bash
uv run --no-sync python scripts/visualization/build_public_demo.py \
  --showcase-directory outputs/evaluation_showcase/my-showcase \
  --output-directory outputs/evaluation_showcase/my-public-demo
```

The allowlist is [public-demo-selection.json](../../evaluation/public-demo-selection.json).
The builder verifies saved checksums and source identity, refuses private-inclusive
showcases and copies only selected pages/assets. It preserves the original saved
payloads/page numbers in excerpts and creates a separate subset PDF for viewing.
IRENA attribution is required; separately credited third-party photos are excluded.
NASA's Apollo photograph is explicitly credited to NASA; NPS marks the diary asset
public domain and its requested collection credit is retained.

[Pages deployment](../../.github/workflows/publish-demo.yml) downloads the release
asset pinned by [digest](../../evaluation/public-demo-bundle.json), verifies its
file inventory and publishes static files. Full private/local archives stay out
of Git, release assets and Pages. Screenshot inputs and hashes are in
[gallery provenance](../assets/readme-gallery.provenance.json).
