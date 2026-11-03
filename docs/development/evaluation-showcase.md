# Browse saved parsing evidence

**[Open the public demo](https://bmaged23.github.io/multimodal-document-parser/?v=5.1)**
or [download the portable ZIP](https://github.com/bmaged23/multimodal-document-parser/releases/download/demo-v5.1/public-demo-v5.1.zip).
It contains 53 selected public/author-provided pages, all four OCR profiles,
and IRENA/IMF OCR/enriched/source-reviewed stages. It runs no live model requests.

## Run the downloaded demo

After downloading `public-demo-v5.1.zip`, use Python 3's standard library only:

```bash
python3 -m zipfile -e public-demo-v5.1.zip public-demo
python3 -m http.server 8765 --bind 127.0.0.1 --directory public-demo
```

Open `http://localhost:8765/`. Serve the files over HTTP rather than opening
`index.html` as a local file, because the viewer fetches its saved JSON assets.
No project environment, model download, GPU or cloud key is needed for this path.

## Public demo versus the full local explorer

The public bundle contains 18 NASA pages, 15 IRENA pages, 16 IMF statistical-table
pages and all four author-written assignment pages. Separately credited third-party photo pages and
IMF narrative pages stay out of the public bundle; the full saved selections
remain local. The public bundle is a curated selection: original page mappings, failures and
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

All four documents appear together in the sidebar. Use **Expand PDF + output**,
located beside the reader heading, to give both panes the window. A document
selector remains available while expanded. Enter an original PDF page number
to jump; arrows skip pages outside the published selection. Output formats use
one dropdown, while metrics and display settings stay collapsed until requested.
The full PDF is fetched only when you choose **PDF viewer**; page-image mode
avoids that extra download. Drag the divider
to resize them; zoom the source and output independently. Comparison uses model
or stage tabs by default, with optional side-by-side output. Page navigation
synchronizes the source and extraction and retains original publication pages.
Content, Markdown preview/source, canonical JSON and saved raw replies are separate.
Source-region buttons highlight the corresponding PDF area.

The default public-source selection includes NASA, the author’s handwritten mathematics assignment,
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
NASA's Apollo photograph is explicitly credited to NASA. The assignment is
original work supplied by its author for this public demo. Earlier diary assets
and their NPS collection attribution remain in versioned historical bundles.

[Pages deployment](../../.github/workflows/publish-demo.yml) downloads the release
asset pinned by [digest](../../evaluation/public-demo-bundle.json), verifies its
file inventory and publishes static files. Full private/local archives stay out
of Git, release assets and Pages. Screenshot inputs and hashes are in
[gallery provenance](../assets/readme-gallery.provenance.json).

## Hosted validation

The [public demo browser check](../../evaluation/results/public-demo-validation-v3.json)
verified all three model selectors, the five output formats, original page mapping,
failed outputs, pipeline-stage source highlights, all hosted subset-PDF hashes,
expanded desktop geometry and mobile overflow. The full local suite passed 140
tests, including five public-excerpt isolation/integrity tests.
