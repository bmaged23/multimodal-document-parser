"""Render an implementation-checked animated data-flow diagram and clickable-demo banner."""

import ast
import hashlib
import json
import math
import subprocess
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD = FONT.with_name("DejaVuSans-Bold.ttf")
NODES = [
    ("input", (40, 190, 250, 335), "PDF or DOCX", ["PDF input", "DOCX via LibreOffice"], "#087e71"),
    (
        "pages",
        (305, 190, 535, 335),
        "Render pages",
        ["Page images + identity", "Page geometry retained"],
        "#087e71",
    ),
    (
        "ocr",
        (590, 190, 825, 335),
        "dots.mocr Q4_K_M",
        ["Q4 language + Q8 projector", "Tables + visual regions"],
        "#087e71",
    ),
    (
        "json",
        (880, 190, 1120, 335),
        "Normalize + crop",
        ["Canonical JSON + boxes", "Page and visual assets"],
        "#087e71",
    ),
    (
        "exports",
        (1260, 190, 1560, 335),
        "Review + export",
        ["JSON · Markdown · HTML", "Source PDF + linked assets"],
        "#087e71",
    ),
    (
        "evidence",
        (335, 495, 765, 650),
        "Keep the evidence",
        ["Raw OCR / VLM replies + logs", "Failures and partial runs retained"],
        "#5e7685",
    ),
    (
        "vlm",
        (870, 470, 1245, 660),
        "Optional visual enrichment",
        [
            "Crops + page context",
            "Gemma4:31b via Ollama Cloud",
            "Schema-check replies; keep provenance",
            "Generated descriptions + chart values",
        ],
        "#466bad",
    ),
]
EDGES = [
    ("input", "pages", [(250, 260), (305, 260)], "#087e71"),
    ("pages", "ocr", [(535, 260), (590, 260)], "#087e71"),
    ("ocr", "json", [(825, 260), (880, 260)], "#087e71"),
    ("json", "exports", [(1120, 260), (1260, 260)], "#087e71"),
    ("json", "vlm", [(1000, 335), (1000, 470)], "#466bad"),
    ("vlm", "exports", [(1245, 565), (1410, 565), (1410, 335)], "#466bad"),
    ("ocr", "evidence", [(707, 335), (707, 410), (550, 410), (550, 495)], "#5e7685"),
    ("vlm", "evidence", [(870, 570), (765, 570)], "#5e7685"),
]


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def flow_point(points, fraction):
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:], strict=False)]
    remaining = fraction * sum(lengths)
    for a, b, length in zip(points, points[1:], lengths, strict=False):
        if remaining <= length:
            ratio = remaining / length
            return a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio
        remaining -= length
    return points[-1]


def arrow(draw, points, color):
    draw.line(points, fill=color, width=4, joint="curve")
    end, before = points[-1], points[-2]
    angle = math.atan2(end[1] - before[1], end[0] - before[0])
    triangle = [end]
    for offset in [-0.55, 0.55]:
        triangle.append(
            (end[0] - 13 * math.cos(angle + offset), end[1] - 13 * math.sin(angle + offset))
        )
    draw.polygon(triangle, fill=color)


def verify_implementation():
    contracts = {
        "src/multimodal_doc_parser/pipeline_services/stage_handlers.py": [
            "convert_docx",
            "execute_stage",
        ],
        "src/multimodal_doc_parser/pdf_preparation/prepare_document.py": ["prepare_document"],
        "src/multimodal_doc_parser/parser_adapters/parsing_run.py": ["parse_prepared_document"],
        "src/multimodal_doc_parser/parser_adapters/dots_mocr_adapter.py": [
            "normalize_dots_mocr_results"
        ],
        "src/multimodal_doc_parser/visual_enrichment/ollama.py": ["enrich_document"],
        "src/multimodal_doc_parser/document_exports/readable_exports.py": ["document_markdown"],
        "src/multimodal_doc_parser/review_interface/html_report.py": ["write_html_report"],
    }
    hashes = {}
    for relative, names in contracts.items():
        source = (ROOT / relative).read_text()
        definitions = {
            n.name
            for n in ast.walk(ast.parse(source))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert set(names) <= definitions, relative
        hashes[relative] = hashlib.sha256(source.encode()).hexdigest()
    stage = (ROOT / "src/multimodal_doc_parser/pipeline_services/stage_handlers.py").read_text()
    assert "if parsing.enrich_visuals:" in stage and "output = ocr_output" in stage
    assert 'provider="ollama-cloud", model="gemma4:31b"' in stage
    parsing = (ROOT / "src/multimodal_doc_parser/parser_adapters/parsing_run.py").read_text()
    for token in [
        "normalize_dots_mocr_results",
        "write_html_report",
        "document_markdown",
        '"inference.log"',
        '"raw"',
    ]:
        assert token in parsing
    return hashes


def main():
    hashes = verify_implementation()
    ASSETS.mkdir(parents=True, exist_ok=True)
    base = Image.new("RGB", (1600, 810), "#f3f7f8")
    draw = ImageDraw.Draw(base)
    draw.text((40, 35), "HOW THE PARSING PIPELINE WORKS", font=font(31, True), fill="#173441")
    draw.text(
        (40, 90),
        "The deployed quantized dots path · local GPU + optional cloud visual descriptions",
        font=font(22),
        fill="#54717e",
    )
    for _, _, points, color in EDGES:
        arrow(draw, points, color)
    draw.text((1143, 225), "OCR only", font=font(16), fill="#087e71")
    draw.text((1018, 390), "Crops + context", font=font(17), fill="#466bad")
    draw.text((1280, 595), "Enriched output", font=font(17), fill="#466bad")
    draw.text((440, 435), "Raw predictions", font=font(16), fill="#5e7685")
    draw.text((775, 535), "Raw reply", font=font(14), fill="#5e7685")
    for _, bounds, title, lines, color in NODES:
        x0, y0, x1, y1 = bounds
        draw.rounded_rectangle(bounds, radius=17, fill="white", outline=color, width=3)
        draw.rounded_rectangle((x0, y0, x1, y0 + 8), radius=4, fill=color)
        size = 23 if x1 - x0 > 300 else 20
        draw.text((x0 + 16, y0 + 24), title, font=font(size, True), fill="#173441")
        for index, line in enumerate(lines):
            draw.text((x0 + 16, y0 + 66 + index * 25), line, font=font(16), fill="#4d6876")
    draw.text(
        (40, 724),
        (
            "Arrows illustrate data flow, not processing speed. OCR-only output "
            "bypasses the cloud step."
        ),
        font=font(19),
        fill="#3c5c69",
    )
    draw.text(
        (40, 762),
        (
            "Raw-model comparison and evaluation-only source corrections are separate"
            " from this runtime flow."
        ),
        font=font(18),
        fill="#5b7480",
    )
    base.save(ASSETS / "parsing-pipeline-poster.png")
    frames = []
    for number in range(48):
        frame = base.copy()
        pen = ImageDraw.Draw(frame)
        for index, (_, _, points, color) in enumerate(EDGES):
            x, y = flow_point(points, (number / 48 + index * 0.13) % 1)
            pen.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=2)
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=96))
    frames[0].save(
        ASSETS / "parsing-pipeline.gif",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=True,
    )
    # Static vector alternative for readers who prefer no animation.
    svg = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="810" '
            'viewBox="0 0 1600 810"><title>Verified document parsing '
            'pipeline</title><rect width="1600" height="810" fill="#f3f7f8"/>'
        )
    ]
    svg.append(
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M '
        '0 0 L 10 5 L 0 10 z" fill="#087e71"/></marker></defs>'
    )
    svg.append(
        '<g font-family="DejaVu Sans,Arial,sans-serif" fill="#173441"><text '
        'x="40" y="65" font-size="31" font-weight="700">HOW THE PARSING PIPELINE '
        'WORKS</text><text x="40" y="112" font-size="22">Deployed quantized dots path:'
        " local GPU and optional Ollama Cloud visual descriptions</text>"
    )
    for _, _, points, color in EDGES:
        coordinates = " ".join(f"{x},{y}" for x, y in points)
        svg.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="4" marker-end="url(#arrow)"/>'
        )
    for _, (x0, y0, x1, y1), title, lines, color in NODES:
        svg.append(
            f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" '
            f'rx="17" fill="white" stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{x0 + 16}" y="{y0 + 47}" font-size="20" font-weight="700">'
            f"{escape(title)}</text>"
        )
        for index, line in enumerate(lines):
            svg.append(
                f'<text x="{x0 + 16}" y="{y0 + 84 + index * 25}" font-size="16">'
                f"{escape(line)}</text>"
            )
    svg.append(
        '<text x="40" y="747" font-size="19">Optional cloud branch; raw replies '
        "and failures retained. Source-reviewed evaluation repairs are "
        "separate.</text></g></svg>"
    )
    (ASSETS / "parsing-pipeline.svg").write_text("".join(svg))
    banner = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="165" '
        'viewBox="0 0 1200 165"><title>Open the interactive PDF parsing '
        'demo</title><rect width="1200" height="165" rx="20" '
        'fill="#123844"/><circle cx="80" cy="82" r="41" fill="#19b99a"/><path '
        'd="M69 59L101 82L69 105Z" fill="white"/><g font-family="DejaVu '
        'Sans,Arial,sans-serif"><text x="148" y="64" font-size="30" font-'
        'weight="700" fill="white">Interactive PDF Parsing Demo</text><text '
        'x="148" y="106" font-size="19" fill="#c3e4e4">NASA · IRENA charts · IMF '
        'tables · handwriting</text><rect x="938" y="48" width="222" height="68" '
        'rx="12" fill="#22c6a4"/><text x="975" y="91" font-size="23" font-'
        'weight="700" fill="#092f32">OPEN DEMO →</text></g></svg>'
    )
    (ASSETS / "interactive-demo-banner.svg").write_text(banner)
    release = json.loads(
        (ROOT / "evaluation/results/container-release-validation-v1.json").read_text()
    )["automated_release"]
    revision = release["source_revision_verified"]
    released_hashes = {
        relative: hashlib.sha256(
            subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
        ).hexdigest()
        for relative in hashes
    }
    record = {
        "scope": (
            "Deployed quantized dots parsing path, not the separate RAG pipeline or "
            "evaluation-only source corrections"
        ),
        "review": (
            "Input preparation, OCR-only bypass, optional Gemma cloud branch, raw "
            "evidence retention and export calls checked against implementation"
        ),
        "implementation_sha256": hashes,
        "released_image_source_revision": revision,
        "matches_released_runtime_sources": released_hashes == hashes,
        "nodes": [node[0] for node in NODES],
        "edges": [[edge[0], edge[1]] for edge in EDGES],
        "animated_frames": 48,
        "animation_duration_ms": 4800,
        "assets": {
            name: hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
            for name in [
                "parsing-pipeline.gif",
                "parsing-pipeline.svg",
                "parsing-pipeline-poster.png",
                "interactive-demo-banner.svg",
            ]
        },
    }
    (ASSETS / "pipeline-diagram-verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Generated 48-frame pipeline animation, static alternatives and interactive-demo banner.")


if __name__ == "__main__":
    main()
