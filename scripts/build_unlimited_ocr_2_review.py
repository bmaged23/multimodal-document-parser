"""Create a diagnostic PDF/Markdown review without claiming canonical normalization."""

import argparse
import html
import json
import re
import shutil
from pathlib import Path

from PIL import Image

from multimodal_doc_parser.review_interface.html_report import render_markdown_preview

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"<BLOCK>([A-Za-z_][\w-]*)(\[[^\]\n]*\])<FORK>(.*?)(?=<BLOCK>|\Z)", re.S)
REFERENCE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>", re.S)


def visual_markdown(text: str, source: Path, output: Path, page: int) -> str:
    image = Image.open(source)
    index = 0

    def crop_coordinates(payload: str, scale: int) -> str:
        nonlocal index
        try:
            boxes = json.loads(payload)
            if len(boxes) == 4 and all(isinstance(x, (int, float)) for x in boxes):
                boxes = [boxes]
            references = []
            for box in boxes:
                if len(box) != 4 or not all(
                    isinstance(x, (int, float)) and 0 <= x <= scale for x in box
                ):
                    raise ValueError("Invalid predicted image coordinates")
                x0, y0, x1, y1 = box
                if not x0 < x1 or not y0 < y1:
                    raise ValueError("Empty predicted image box")
                index += 1
                name = f"assets/page_{page:04d}_crop_{index:03d}.png"
                image.crop(
                    (
                        x0 / scale * image.width,
                        y0 / scale * image.height,
                        x1 / scale * image.width,
                        y1 / scale * image.height,
                    )
                ).save(output / name)
                references.append(f"![Model-predicted image crop]({name})")
            return "\n" + "\n".join(references) + "\n"
        except (ValueError, TypeError, json.JSONDecodeError):
            return "\n[Image coordinates invalid; inspect raw output and source page.]\n"

    def replacement(match: re.Match[str]) -> str:
        return crop_coordinates(match[2], 999) if match[1].strip() == "image" else ""

    def block_markdown(match: re.Match[str]) -> str:
        category, coordinates, content = match.groups()
        content = content.strip()
        if category in {"image", "chart"}:
            return crop_coordinates(coordinates, 1000) + content + "\n\n"
        if category == "title":
            return "\n## " + content + "\n\n"
        return content + "\n\n"

    text = text.replace("<PAGE>", "\n")
    if "<BLOCK>" in text:
        return BLOCK.sub(block_markdown, text)
    return REFERENCE.sub(replacement, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    status = json.loads((run / "run-status.json").read_text())
    bundle = (
        Path(status["prepared_bundle"])
        if status.get("prepared_bundle")
        else next((ROOT / "outputs/parsed_documents").glob("be2ebf*-dpi-144"))
    )
    source = json.loads((bundle / "document.json").read_text())
    (run / "assets").mkdir(exist_ok=True)
    shutil.copyfile(bundle / "source.pdf", run / "source.pdf")
    cards = []
    for page in source["pages"]:
        number = page["page_number"]
        prediction = run / f"page_{number:04d}.json"
        if not prediction.exists():
            cards.append(
                f'<section id="page-{number}"><h2>Page {number}</h2><p>No output.</p></section>'
            )
            continue
        raw = json.loads(prediction.read_text())
        markdown = visual_markdown(raw["text"], bundle / page["image"]["path"], run, number)
        (run / f"page_{number:04d}.md").write_text(markdown)
        label = (
            "Finished generation; accuracy unscored"
            if raw["finish_reason"] == "stop"
            else "INCOMPLETE generation"
        )
        cards.append(
            f'<section id="page-{number}"><h2>Page {number}</h2><p>{label}</p>'
            f"<div>{render_markdown_preview(markdown)}</div>"
            "<details><summary>Markdown source</summary>"
            f"<pre>{html.escape(markdown)}</pre></details>"
            "<details><summary>Raw model output</summary>"
            f"<pre>{html.escape(raw['text'])}</pre></details></section>"
        )
    error = html.escape(status.get("error", ""))
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Unlimited-OCR 2 — experimental GPU review</title><style>
body{margin:0;font:16px system-ui;color:#e4e9f2;background:#111827}
header{padding:14px;background:#1e293b}
main{display:grid;grid-template-columns:1fr 1fr;height:calc(100vh - 140px)}
iframe{width:100%;height:100%;border:0}
article{overflow:auto;padding:20px}section{border-bottom:1px solid #64748b;padding-bottom:30px}
img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #64748b;padding:5px}
pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:6px;margin:5px}
</style><header><b>Unlimited-OCR 2 · NF4 GPU experiment</b>
<p>Diagnostic output, not normalized project JSON. Predicted crops may be incomplete.
No accuracy score. Status: STATUS</p>ERROR
<label>Page <select onchange="go(Number(this.value))">PAGE_OPTIONS</select></label></header>
<main><iframe title="Original source PDF" id="pdf" src="source.pdf#page=1"></iframe>
<article>CARDS</article></main>
<script>function go(n){document.getElementById('pdf').src='source.pdf#page='+n;
document.getElementById('page-'+n).scrollIntoView({block:'start'});}</script></html>"""
    document = document.replace(
        "PAGE_OPTIONS",
        "".join(
            f'<option value="{p["page_number"]}">{p["page_number"]}</option>'
            for p in source["pages"]
        ),
    )
    document = document.replace("STATUS", html.escape(status["status"]))
    document = document.replace("ERROR", f"<p>{error}</p>" if error else "")
    (run / "review.html").write_text(document.replace("CARDS", "".join(cards)))
    print(run / "review.html")


if __name__ == "__main__":
    main()
