# Document schema v1

Optional quality metadata added after the IRENA/IMF review remains backward-readable
under schema 1.0. Table cells expose `header_inferred`; tables can record the
header-inference method, source caption/note IDs, logical table ID, continuation
index and column group. Caption/note references are validated against the document.
Visual analysis can record `plot_type`; `quality_review` distinguishes unverified,
model-reviewed, source-correlated and needs-review states. Source-extracted numeric
series keep their own provenance, separate from model-generated descriptions and
source-based endpoint summaries. These fields describe evidence, not a general
accuracy guarantee.

The canonical schema is implemented in
`src/multimodal_doc_parser/document_schema/document.py`. The exported
`schemas/document-v1.schema.json` describes field types. Pydantic also checks
cross-field and reference constraints that JSON Schema alone does not express.

## Identity, preparation, and assets

`document_id` is `sha256:` followed by the complete hash of the input PDF bytes.
Renaming a PDF or rendering it at another resolution does not change its identity.
A preparation bundle is named `<sha256>-dpi-<dpi>` and contains:

```text
source.pdf                       Snapshot of the exact input bytes
document.json                    Validated document, initially status=prepared
assets/pages/page_0001.png        Rendered page 1
assets/pages/page_0002.png        Rendered page 2
```

Paths in JSON are POSIX paths relative to the bundle. Absolute paths and parent
traversal are rejected. Inputs and outputs stay ignored by Git. A renderer version
is recorded for reproducibility. Existing bundles are never overwritten; use a
new output directory to rerun the same preparation settings.

Preparation creates an empty `elements` array. This does not mean the document
contains no text: no parsing has happened yet. Parser adapters will populate
`elements` and set `status=parsed`. Live exports use `status=parsing` while
only some pages have extracted elements; unfilled pages are still pending. Markdown export starts in milestone 3.

## Geometry and order

- Pages are 1-based, contiguous, and in source order.
- `width_points` and `height_points` describe the displayed page in PDF points
  (72 per inch), after intrinsic rotation. Pixel dimensions describe the saved PNG.
- Bounding boxes use normalized coordinates from 0 to 1, with a top-left origin
  on the displayed, rotated page. Each box has positive width and height.
- Pixel conversion: multiply x by page pixel width and y by page pixel height.
  Adapter code must transform backend coordinates into this convention.
- `rotation_degrees` records the original page rotation already applied by rendering.
  Consumers must not rotate the image again.
- The `elements` array is the canonical reading order across all pages.
- `parent_id` references an earlier heading. A nested heading must have a greater
  level than its parent. This prevents cycles and preserves logical sections.
- An element may have multiple source regions, allowing tables or paragraphs
  spanning multiple pages. Page headers and footers have their own element types.

## Content and provenance

The discriminated element types cover text, list items, captions, footnotes,
headers, footers, headings, formulas (LaTeX), tables, charts, and images.

Tables store 0-based cell coordinates and row/column spans. Overlapping cells and
spans outside the declared grid are rejected. Missing cells are allowed to preserve
incomplete parser output rather than invent content. Blank observed cells should
be represented explicitly with empty text.

Visual elements reference source crops. Descriptions have separate provenance
from the extracted element. Chart points distinguish `printed_label`,
`visual_estimate`, and `unknown`; unknown points must have null values. Values
must be finite. Each series can record its unit. More complex chart representations
can evolve through a versioned schema as real backend outputs are evaluated.

All elements record their producer, optional model version, and whether their
content was extracted, generated, or manually supplied. Confidence scores are
omitted until their meanings can be defined per backend.

`example-parsed-document.json` is a synthetic schema illustration, not model
output. Its asset paths and all-zero document hash are placeholders; corresponding
assets are intentionally not shipped. Tests generate real preparation bundles.

## Scope and limits

Preparation renders one page at a time using PDFium, closes native resources,
and publishes a bundle only after every page succeeds. Invalid/encrypted PDFs
are rejected; password support is not implemented. Default limits are 200 pages
and 25 million rendered pixels per page at 144 DPI. These limits do not constitute
a complete security boundary for a public upload service; service-level upload
limits, process isolation, timeouts, and retention policy belong to deployment.

PDFium is not thread-safe. In-process rendering calls must be serialized; future
parallel workers must use separate processes.

Implementation references:
[Pydantic models](https://docs.pydantic.dev/latest/concepts/models/) and
[PDFium wrapper API](https://pypdfium2.readthedocs.io/en/stable/python_api.html).
