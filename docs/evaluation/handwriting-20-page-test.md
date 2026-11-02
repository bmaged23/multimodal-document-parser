# Twenty-page handwriting and mixed-content test

This case uses real scans from the NPS Lucinda Davis Diary (1874), not generated
handwriting or handwriting fonts. The versioned selection and original-page map
are in `evaluation/test_cases/lucinda-davis-handwriting-20.json`.

## Selected source PDF pages

4, 5, 7, 11, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 68, 74, 76.

These are 20 PDF pages, mostly containing two-page book spreads. The selection
covers 18 pages containing handwriting, including mixed-content pages, and two
printed reference pages:

- Pages 5 and 7: printed calendar/astronomical and reference tables.
- Pages 4 and 11: mixed print and handwriting.
- Page 76: handwritten numerical ledger; it is not guaranteed to be detected as
  a structured table and must be assessed separately from narrative text.
- Other selected pages: historical cursive diary entries and memoranda, including
  different writing density, faint ink, curved spreads, and sparse entries.

Visual inspection of the source found no genuine plotted charts. Calendars,
reference tables, and numerical lists are not labeled as charts. This test does
not establish chart extraction quality. All source pages lack a PDF text layer.

## Configuration

- Model: complete PaddleOCR-VL-1.6 pipeline.
- GPU model instances: one; page window: one.
- Rendering: **300 DPI**, producing approximately 1413 × 1244 pixels for the
  first selected spreads. The physically small PDF page dimensions would make
  the faint handwriting too small at the earlier 144-DPI setting.
- Orientation classification and unwarping remain disabled to preserve alignment.
- Inputs are unchanged source-page imports; no artificial graphs or handwriting
  have been added. Original source and subset SHA-256 hashes are recorded.
- The NASA parsing run remains stopped. This test runs in its own directory.

Output directory:
`outputs/parser_runs/lucinda-davis-handwriting-20-paddleocr-vl-1.6/`.

The detached controller writes a per-page inference manifest under `raw/paddleocr/`.
`document.json`, `document.md`, and `review.html` are exported after the full test
finishes. No live preview server is started. Until completion, check the inference
manifest rather than treating the prepared document as parsed output.

## Review protocol

After inference, inspect whether the layout detector actually routed handwritten
regions to text recognition, rather than only preserving them as image crops.
Check legible words, dates, numbers, table rows/columns, reading order between
left/right pages, omitted faint lines, and unsupported invented text. Record
unreadable reference regions explicitly. Printed-table quality must not be
reported as handwritten-text accuracy.

No verified transcription is available yet, so no CER/WER or numeric accuracy
score is claimed. A manually verified sample will be needed before scoring.
This is one historical writer and does not establish modern handwriting or
multi-writer performance. Results are pending.

Source and attribution:
[NPS diary record](https://npgallery.nps.gov/AssetDetail/539a8c6d-d1c2-40ec-9729-ba874f53ccb1).
The source marks the asset public domain and requests credit: Courtesy of North
Cascades NPS Complex Museum Collection, Aerial Photograph Collection.
