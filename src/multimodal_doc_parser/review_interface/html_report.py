"""Build an offline report with synchronized PDF and Markdown inspection."""

import json
import re
from html import escape
from importlib.resources import files
from pathlib import Path

import nh3
from markdown_it import MarkdownIt

from multimodal_doc_parser.document_exports.readable_exports import (
    document_markdown,
    element_html,
)
from multimodal_doc_parser.document_schema.document import Document


def render_markdown_preview(markdown: str) -> str:
    """Render Markdown, including generated table HTML, then sanitize the result."""
    rendered = MarkdownIt("commonmark", {"html": True}).enable("table").render(markdown)
    cleaned = nh3.clean(
        rendered,
        tags={
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "br",
            "hr",
            "em",
            "strong",
            "s",
            "sup",
            "details",
            "summary",
            "blockquote",
            "pre",
            "code",
            "ul",
            "ol",
            "li",
            "a",
            "img",
            "table",
            "thead",
            "tbody",
            "tfoot",
            "tr",
            "th",
            "td",
            "caption",
            "div",
        },
        attributes={
            "a": {"href", "title"},
            "img": {"src", "alt", "title"},
            "th": {"rowspan", "colspan"},
            "td": {"rowspan", "colspan"},
            "div": {"class"},
            "code": {"class"},
            "ol": {"start"},
        },
        url_schemes={"http", "https", "mailto"},
    )
    return cleaned.replace("<img ", '<img loading="lazy" decoding="async" ')


def write_html_report(document: Document, bundle: Path, warnings: list[str]) -> Path:
    cards = []
    for element in document.elements:
        region = element.source_regions[0]
        box = region.bounding_box
        coordinates = f"{box.x0},{box.y0},{box.x1},{box.y1}"
        pages = ",".join(str(item.page_number) for item in element.source_regions)
        cards.append(
            f'<article class="element-card" id="{escape(element.id, quote=True)}" '
            f'data-page="{region.page_number}" data-pages="{pages}" '
            f'data-box="{coordinates}" tabindex="0">'
            f'<div class="element-meta"><span>{escape(element.type.replace("_", " "))}</span>'
            f'<button class="source-link" type="button">Page {region.page_number} ↗</button>'
            f"</div>{element_html(element)}</article>"
        )
    if not cards:
        cards.append('<div class="empty-state">No extracted elements in this document.</div>')
    markdown_pages = []
    source_pages = []
    json_pages = []
    for page in document.pages:
        number = page.page_number
        markdown = document_markdown(document, page_number=number)
        markdown_pages.append(
            f'<section class="markdown-page" data-page="{number}">'
            f'<div class="page-heading">Page {number}</div>{render_markdown_preview(markdown)}'
            "</section>"
        )
        source_pages.append(
            f'<pre class="raw-output source-page" data-page="{number}">{escape(markdown)}</pre>'
        )
        page_json = {
            "document_id": document.document_id,
            "page": page.model_dump(),
            "elements": [
                element.model_dump()
                for element in document.elements
                if any(region.page_number == number for region in element.source_regions)
            ],
        }
        json_pages.append(
            f'<pre class="raw-output json-page" data-page="{number}">'
            f"{escape(json.dumps(page_json, indent=2, ensure_ascii=False))}</pre>"
        )
    page_images = json.dumps([page.image.path for page in document.pages]).replace("<", "\\u003c")
    template = files("multimodal_doc_parser.review_interface").joinpath("report_template.html")
    progress_notice = ""
    if document.status == "parsing":
        message = (
            "This is a partial extraction; refresh to see completed pages. "
            "Pages without output are still pending."
        )
        status_path = bundle / "run-status.json"
        if status_path.exists():
            state = json.loads(status_path.read_text())
            if state.get("status") in {"completed_with_errors", "failed", "interrupted"}:
                count = state.get("completed_pages", "?")
                total = state.get("total_pages", len(document.pages))
                message = (
                    f"Parsing finished with incomplete output: {count}/{total} pages accepted. "
                    "See the extraction notes and per-page diagnostics."
                )
        progress_notice = '<div class="progress-notice">' + escape(message) + "</div>"
    replacements = {
        "TITLE": escape(document.title or document.source_filename),
        "PROGRESS_NOTICE": progress_notice,
        "FILENAME": escape(document.source_filename),
        "MODEL_LABEL": escape(
            "; ".join(
                sorted(
                    {
                        " ".join(
                            filter(
                                None,
                                [element.provenance.producer, element.provenance.model_version],
                            )
                        )
                        for element in document.elements
                    }
                )
            )
            or "No parser output"
        ),
        "ELEMENT_COUNT": str(len(document.elements)),
        "PAGE_COUNT": str(len(document.pages)),
        "WARNING_COUNT": str(len(warnings)),
        "WARNING_LIST": "".join(f"<li>{escape(item)}</li>" for item in warnings) or "<li>None</li>",
        "CARDS": "".join(cards),
        "MARKDOWN_PREVIEW": render_markdown_preview(document_markdown(document)),
        "MARKDOWN_PAGES": "".join(markdown_pages),
        "MARKDOWN_SOURCE_PAGES": "".join(source_pages),
        "JSON_PAGES": "".join(json_pages),
        "MARKDOWN": escape(document_markdown(document)),
        "JSON": escape(document.model_dump_json(indent=2)),
        "PAGE_IMAGES": page_images,
    }
    html = template.read_text(encoding="utf-8")
    # One pass prevents extracted text from becoming another template token.
    html = re.sub(r"@@([A-Z_]+)@@", lambda match: replacements[match[1]], html)
    destination = bundle / "review.html"
    temporary = bundle / "review.html.tmp"
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(destination)
    return destination
