"""Native Unlimited blocks keep source coordinates and incomplete-page failures."""

import json

from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.parser_adapters.unlimited_saved import normalize_saved_run


def test_native_blocks_keep_merged_cells_and_do_not_fill_failed_pages(showcase_project):
    project, _ = showcase_project
    prepared = next((project / "prepared").iterdir())
    parent = json.loads((prepared / "document.json").read_text())
    source = project / "native"
    source.mkdir()
    (source / "run-status.json").write_text(
        json.dumps(
            {
                "document_id": parent["document_id"],
                "prepared_bundle": str(prepared),
            }
        )
    )
    text = (
        "<PAGE><BLOCK>title[100,50,900,100]<FORK>Own OCR heading"
        "<BLOCK>table[100,150,900,500]<FORK>"
        '<table><tr><th rowspan="2">Region</th><td>12</td></tr>'
        "<tr><td>34</td></tr></table>"
        "<BLOCK>image[100,550,900,900]<FORK>"
    )
    (source / "page_0001.json").write_text(
        json.dumps(
            {
                "finish_reason": "stop",
                "text": text,
                "elapsed_seconds": 1.5,
            }
        )
    )
    (source / "page_0002.json").write_text(
        json.dumps(
            {
                "finish_reason": "length",
                "text": "<PAGE><BLOCK>text[1,1,99,99]<FORK>unfinished",
            }
        )
    )
    output = normalize_saved_run(source, project / "normalized")
    doc = Document.model_validate_json((output / "document.json").read_text())
    assert doc.status == "parsing"
    assert len(doc.elements) == 3
    table = next(e for e in doc.elements if e.type == "table")
    assert table.cells[0].row_span == 2
    assert [c.text for c in table.cells] == ["Region", "12", "34"]
    visual = next(e for e in doc.elements if e.type == "image")
    assert visual.source_regions[0].bounding_box.x0 == 0.1
    assert (output / visual.asset.path).is_file()
    assert all(e.provenance.producer == "Unlimited-OCR 2" for e in doc.elements)
    manifest = json.loads((output / "raw/unlimited_ocr_2/inference-manifest.json").read_text())
    assert manifest["completed_pages"] == 1
    assert manifest["failed_pages"][0]["page_number"] == 2
    assert json.loads((source / "page_0001.json").read_text())["text"] == text
