"""Build a local report for the frozen 70-page manual-reference development study."""

# ruff: noqa: E501 -- Embedded HTML/JavaScript template and metric descriptions.

import argparse
import html
import json
from pathlib import Path

METRICS = [
    (
        "text",
        "character_error_rate",
        "Region character error rate ↓",
        "Edits per reference character, using the best matching span inside overlapping OCR blocks. "
        "Measures scoped recovery; extra text outside the matched span is not penalized.",
    ),
    (
        "text",
        "word_error_rate",
        "Region word error rate ↓",
        "Word edits per reference word with the same source-region matching. Lower is better.",
    ),
    (
        "text",
        "token_f1",
        "Text token F1 ↑",
        "Balances token coverage and extra tokens in geometry-matched text blocks. Order is ignored.",
    ),
    (
        "table",
        "strict_cell_text_match",
        "Strict table cell match ↑",
        "Exact normalized text at the reference grid row and column. Different valid layouts can score poorly.",
    ),
    (
        "table",
        "keyed_row_value_match",
        "Label-linked row/value match ↑",
        "Matches a unique country, objective ID or other row key, then checks its ordered values. "
        "Tolerates header-row offsets; missing or duplicate keys fail.",
    ),
    (
        "table",
        "cell_text_multiset_f1",
        "Table cell content F1 ↑",
        "Compares the multiset of nonempty cell texts. Shows content coverage, but ignores row associations.",
    ),
    (
        "table",
        "span_match",
        "Merged-cell span match ↑",
        "Row-span and column-span attributes matching at exact grid origins. Layout-sensitive.",
    ),
    (
        "table",
        "header_role_match",
        "Table header role match ↑",
        "Correct explicit header/body flags at exact grid origins. No inferred pipeline flags are added.",
    ),
    (
        "visual",
        "region_recall_at_iou_0_5",
        "Image/chart/diagram region recall ↑",
        "Fraction of annotated visuals with a unique OCR region overlapping at least 50% IoU. "
        "Includes small inset photos and QR codes; logos/decorations are excluded.",
    ),
    (
        "visual",
        "matched_mean_iou",
        "Matched crop overlap ↑",
        "Mean intersection-over-union among matched visual regions. Does not penalize missing regions; "
        "read alongside region recall.",
    ),
    (
        "heading",
        "heading_detection_recall",
        "Heading detection recall ↑",
        "Reference headings matched to native title/heading blocks at IoU ≥ 0.30. "
        "Does not score heading levels or parent links.",
    ),
    (
        "order",
        "matched_block_pair_order",
        "Matched text reading order ↑",
        "Correctly ordered pairs among source text blocks that could be matched geometrically. "
        "Unmatched blocks are outside this denominator.",
    ),
    (
        "chart",
        "printed_numeric_token_recall",
        "Chart numeric-token recovery ↑",
        "Printed chart values found in raw OCR text overlapping the figure. "
        "Axis values may create matches; this is not series/category understanding.",
    ),
]
LABELS = dict(
    paddle="PaddleOCR-VL-1.6",
    dots="dots.mocr Q4/Q8",
    bf16="dots.mocr original BF16",
    unlimited="Unlimited-OCR 2 NF4/FP16",
)


def build(directory, demo_directory=None):
    data = json.loads((directory / "metrics.json").read_text())
    pages = json.loads((directory / "page-metrics.json").read_text())
    excluded = json.loads((directory / "excluded-references.json").read_text())
    if demo_directory:
        catalog = json.loads((demo_directory / "catalog.json").read_text())
        routes = dict(
            nasa="nasa-financial", math="handwriting", irena="irena-pipeline", imf="imf-pipeline"
        )
        for row in pages:
            case = next(c for c in catalog["cases"] if c["id"] == routes[row["case"]])
            original = row["page"] + 12 if row["case"] == "irena" else row["page"]
            row["source_url"] = (
                f"../?case={case['id']}&page={original}&run={row['case']}-{row['profile']}-gemma"
                if original in case["original_pages"]
                else None
            )
    payload = json.dumps(
        dict(
            metrics=data,
            pages=pages,
            excluded=excluded,
            definitions=METRICS,
            labels=LABELS,
            public=bool(demo_directory),
        ),
        ensure_ascii=False,
    )
    payload = payload.replace("<", "\\u003c")
    template = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Four-profile OCR reference evaluation</title>
<style>body{font:15px/1.5 system-ui;color:#173248;background:#f2f6fa;margin:0}header{background:#173248;color:white;padding:24px}header h1{margin:0;font-size:25px}main{max-width:1600px;margin:auto;padding:22px}select,button{font:inherit;padding:8px;border:1px solid #aebfcd;border-radius:6px;background:white}a{color:#1266a0}header a{color:#b5e0ff}.notice{padding:14px;background:#fff3d1;border-left:4px solid #cb9a1e}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}.table-scroll{overflow:auto;background:white}table{border-collapse:collapse;width:100%;min-width:1000px}th,td{border:1px solid #cfdae3;padding:10px;text-align:left;vertical-align:top}thead th{background:#e5eef6}.meaning{min-width:300px;width:34%;font-size:13px}.number{white-space:nowrap}h2{font-size:21px}details{margin:15px 0;padding:12px;background:white}summary{cursor:pointer;font-weight:600}code{overflow-wrap:anywhere}#page-table{font-size:13px}.muted{font-size:13px;color:#48637a}</style>
<header><h1>OCR evaluation against source references</h1><p>70 pages · four execution profiles · 280 page/profile slots</p><a href="http://localhost:8770/" target="_blank">Open source and reference annotations ↗</a></header>
<main><p class="notice"><strong>Development evaluation:</strong> references were created by the assistant from source pages. The user inspected examples and endorsed the method; these are not independently human-annotated labels. Unresolved handwriting is excluded. Raw OCR only: Gemma and pipeline repairs receive no credit.</p>
<div class="controls"><label>Document <select id="case"><option value="all">All four PDFs</option><option value="nasa">NASA — 20 pages</option><option value="math">Handwriting — 4 pages</option><option value="irena">IRENA — 20 pages</option><option value="imf">IMF — 26 pages</option></select></label><a href="metrics.json">Metrics JSON</a><a href="page-metrics.json">Page metrics</a><a href="failure-traces.json">Detailed checks</a></div>
<p id="coverage"></p><div class="table-scroll"><table id="summary"></table></div>
<p class="muted">↑ higher is better; ↓ lower is better. — means there is no applicable reference denominator. Overall rates pool their underlying units; per-page macro averages are retained in metrics.json. No single combined winner score is calculated.</p>
<details><summary>Scope, matching rules and limitations</summary><ul id="limits"></ul><p>Reference SHA-256: <code id="reference-hash"></code></p><p>Scorer SHA-256: <code id="scorer-hash"></code></p></details>
<details><summary>Excluded annotations and uncertainty</summary><p id="exclusions"></p><a href="excluded-references.json">Open the full exclusion ledger</a></details>
<h2>Page-by-page results</h2><p class="muted">Reference links use selected-PDF page numbers. CER is scoped source-region content recovery; native output presence and usable layout blocks are shown separately. Unlimited's two historical handwriting failures remain in the denominator.</p><div class="table-scroll"><table id="page-table"></table></div></main>
<script type="application/json" id="data">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent),profiles=['paddle','dots','bf16','unlimited'];
const $=id=>document.getElementById(id),pct=x=>x===null||x===undefined?'—':(100*x).toFixed(2)+'%';
function cell(row,value,tag='td'){const el=document.createElement(tag);el.textContent=value;row.append(el);return el;}
function render(){const key=$('case').value,results=key==='all'?data.metrics.summary:data.metrics.per_pdf[key],table=$('summary');table.replaceChildren();const head=table.createTHead().insertRow();cell(head,'Metric','th');profiles.forEach(p=>cell(head,data.labels[p],'th'));cell(head,'What this metric means','th').className='meaning';const body=table.createTBody();for(const [family,name,label,meaning] of data.definitions){const r=body.insertRow();cell(r,label);profiles.forEach(p=>cell(r,pct(results[p][family]?.[name])).className='number');cell(r,meaning).className='meaning';}const slots=results.paddle.page_slots;$('coverage').textContent=profiles.map(p=>`${data.labels[p]}: ${results[p].native_present}/${slots} native outputs, ${results[p].structured_native_pages}/${slots} pages with native layout blocks`).join(' · ');const pt=$('page-table');pt.replaceChildren();const ph=pt.createTHead().insertRow();['Source reference','Profile','Native output','Layout blocks','Region CER ↓','Strict cells ↑','Keyed rows ↑','Visual recall ↑'].forEach(t=>cell(ph,t,'th'));const pb=pt.createTBody();for(const page of data.pages.filter(p=>key==='all'||p.case===key)){const r=pb.insertRow(),td=r.insertCell(),a=document.createElement('a');a.href=data.public?(page.source_url||'#'):`http://localhost:8770/?document=${page.case}&page=${page.page}`;if(data.public&&!page.source_url){a.removeAttribute('href');a.title='Outside the published source excerpt';}a.target='_blank';a.textContent=`${page.case.toUpperCase()} page ${page.page}`;td.append(a);cell(r,data.labels[page.profile]);cell(r,page.native_present?'Present':'Missing');cell(r,String(page.native_blocks));cell(r,pct(page.text?.character_error_rate));cell(r,pct(page.table?.strict_cell_text_match));cell(r,pct(page.table?.reference_keyed_rows?page.table.keyed_row_matches/page.table.reference_keyed_rows:null));cell(r,pct(page.visual?.region_recall_at_iou_0_5));}}
for(const t of data.metrics.limitations){const li=document.createElement('li');li.textContent=t;$('limits').append(li);}$('reference-hash').textContent=data.metrics.reference_sha256;$('scorer-hash').textContent=data.metrics.scorer_sha256;$('exclusions').textContent=`${data.excluded.length} reference annotations excluded from prose CER/WER. This includes separate chart/diagram label blocks and unresolved/deleted handwriting. See the ledger for each reason.`;$('case').onchange=render;render();
</script></html>"""
    if demo_directory:
        template = (
            template.replace('href="http://localhost:8770/"', 'href="../"')
            .replace(
                "Open source and reference annotations ↗", "Open the four-model interactive demo ↗"
            )
            .replace(
                '<a href="failure-traces.json">Detailed checks</a>',
                '<a href="https://github.com/bmaged23/multimodal-document-parser/blob/main/docs/evaluation/README.md">Evaluation methodology</a>',
            )
        )
        template = template.replace("http://localhost:8770/", "../")
    ranking = (
        '<section style="padding:14px;background:white;margin:16px 0;border:1px solid #cfdae3">'
        "<strong>Overall interpretation for this corpus</strong>"
        "<ol><li>dots.mocr BF16 — strongest balance; 91.67% visual recall, 80.96% row/value match.</li>"
        "<li>dots.mocr Q4/Q8 — lowest scoped text error, 0.67%; smaller-GPU production choice.</li>"
        "<li>PaddleOCR-VL-1.6 — best row/value match, 86.58%, and heading detection.</li>"
        "<li>Unlimited-OCR 2 — 55.62% row/value match and two missing handwriting outputs.</li></ol>"
        '<p class="muted">Qualitative judgment across metrics, not a combined score. '
        "The tested Q4/Q8 profile runs on the 6 GB GPU; BF16 reached 8.53 GiB peak PyTorch "
        "allocation on Colab T4, which is not total device memory or a minimum VRAM requirement.</p></section>"
    )
    template = template.replace('<div class="controls">', ranking + '<div class="controls">', 1)
    (directory / "index.html").write_text(template.replace("PAYLOAD", payload))
    print(html.escape(str(directory / "index.html")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--demo-directory", type=Path)
    args = parser.parse_args()
    build(args.directory, args.demo_directory)
