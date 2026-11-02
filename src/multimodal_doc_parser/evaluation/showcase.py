"""Build a portable, read-only showcase from explicitly selected saved artifacts."""

import json
import shutil
import statistics
import tempfile
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from multimodal_doc_parser.document_exports.readable_exports import document_markdown, element_html
from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.evaluation.parser_comparison import load_inference
from multimodal_doc_parser.evaluation.rag_benchmark import digest, save
from multimodal_doc_parser.review_interface.html_report import render_markdown_preview


class SavedRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    label: str
    stage: Literal["ocr", "enriched", "source_checked", "experimental"]
    kind: Literal["canonical", "unlimited"]
    directory: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]+$")


class ShowcaseCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    title: str
    description: str
    group: Literal["parsers", "pipeline"]
    private: bool
    original_pages: list[int] = Field(min_length=1)
    preview_page: int = Field(default=1, ge=1)
    source_url: str | None
    runs: list[SavedRun] = Field(min_length=1)


class ShowcaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["1.0"]
    cases: list[ShowcaseCase]


def safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing or unsafe showcase asset: " + relative)
    return path


def copy_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


class LocalPreview(HTMLParser):
    """Block automatic external image loads while preserving saved source text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            source = next((value for key, value in attrs if key == "src"), None) or ""
            decoded = unquote(unquote(source))
            if (
                urlsplit(decoded).scheme
                or urlsplit(decoded).netloc
                or decoded.startswith(("/", "\\"))
                or "\\" in decoded
                or ".." in decoded.split("/")
                or ":" in decoded
            ):
                self.parts.append("<span>[External or unsafe image reference omitted]</span>")
                return
        self.parts.append(self.get_starttag_text() or "")

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{name};")


def local_preview(markdown: str) -> str:
    parser = LocalPreview()
    parser.feed(render_markdown_preview(markdown))
    parser.close()
    return "".join(parser.parts)


def page_shell(body: str) -> str:
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="referrer" content="no-referrer">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self';
style-src 'unsafe-inline'; base-uri 'self'; form-action 'none'">
<base href="../"><style>
body{font:14px/1.65 system-ui;color:#233548;margin:20px;overflow-wrap:anywhere}
article{background:white;border:1px solid #dfe7ed;border-radius:12px;padding:16px;margin:0 0 14px}
article:hover{border-color:#149c8a}header{font:11px system-ui;color:#607789;display:flex;
justify-content:space-between;margin-bottom:10px}img{max-width:100%;height:auto}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #dce5ed;padding:7px}
th{background:#edf5fa;text-align:left}.table-scroll{overflow:auto}figure{margin:0}
pre{white-space:pre-wrap}aside{padding:14px;background:#fff4d9;border-radius:10px}
button{font:inherit;background:#edf7f5;border:0;color:#087e71;border-radius:6px;padding:4px 8px}
</style>"""
        + body
        + "</html>"
    )


def raw_page(run: Path, number: int, kind: str) -> str | None:
    if kind == "unlimited":
        path = run / f"page_{number:04}.json"
        return str(json.loads(path.read_text()).get("text", "")) if path.exists() else None
    paths = sorted((run / "raw").glob(f"*/page_{number:04}.json"))
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError("Ambiguous raw page artifact")
    data = json.loads(paths[0].read_text())
    if "content" in data:
        return str(data["content"])
    # Deliberately exclude input paths, request envelopes and credentials.
    if "parsing_res_list" in data:
        blocks = [
            {
                k: v
                for k, v in block.items()
                if k in {"block_label", "block_content", "block_bbox", "block_id", "block_order"}
            }
            for block in data["parsing_res_list"]
        ]
        return json.dumps(blocks, indent=2, ensure_ascii=False)
    return None


def canonical_run(run: Path, target: Path, spec: SavedRun) -> tuple[dict[str, Any], Document]:
    document = Document.model_validate_json(safe_file(run, "document.json").read_bytes())
    target.mkdir(parents=True)
    copy_asset(run / "document.json", target / "document.json")
    (target / "document.md").write_text(document_markdown(document))
    # A downloadable document keeps its referenced assets, not only the previews.
    for page in document.pages:
        copy_asset(safe_file(run, page.image.path), target / page.image.path)
    for element in document.elements:
        asset = getattr(element, "asset", None)
        if asset:
            copy_asset(safe_file(run, asset.path), target / asset.path)
    inference = load_inference(run) or {}
    times = {p["page_number"]: p.get("elapsed_seconds") for p in inference.get("pages", [])}
    accepted = set(times)
    acceptance_known = bool(inference.get("pages")) or inference.get("completed_pages") == 0
    pages = []
    for page in document.pages:
        number = page.page_number
        elements = [
            e for e in document.elements if any(r.page_number == number for r in e.source_regions)
        ]
        missing = acceptance_known and number not in accepted
        markdown = document_markdown(document, page_number=number)
        cards = []
        for element in elements:
            region = next(r for r in element.source_regions if r.page_number == number)
            box = region.bounding_box
            coordinates = f"{box.x0},{box.y0},{box.x1},{box.y1}"
            method = element.provenance.method
            description = getattr(element, "description", None)
            if description:
                method += " / visual " + description.provenance.method
            cards.append(
                f'<article data-box="{coordinates}" data-element="{escape(element.id)}">'
                f"<header><span>{escape(element.type)} · {escape(method)}</span>"
                '<button type="button">Locate in source ↗</button></header>'
                + local_preview(element_html(element))
                + "</article>"
            )
        notice = (
            "<aside>No accepted extraction for this page. "
            "Failed output is not an accurate empty page.</aside>"
            if missing
            else ""
        )
        content = notice + ("".join(cards) or "<p>No extracted elements on this page.</p>")
        folder = target / "pages"
        folder.mkdir(exist_ok=True)
        (folder / f"{number:04}-content.html").write_text(page_shell(content))
        (folder / f"{number:04}-preview.html").write_text(
            page_shell(notice + local_preview(markdown))
        )
        pages.append(
            dict(
                number=number,
                status="not_accepted" if missing else "accepted" if acceptance_known else "unknown",
                seconds=times.get(number),
                elements=len(elements),
                content=f"pages/{number:04}-content.html",
                preview=f"pages/{number:04}-preview.html",
                markdown=markdown,
                raw=raw_page(run, number, spec.kind),
                json=dict(
                    document_id=document.document_id,
                    page=page.model_dump(),
                    elements=[e.model_dump(mode="json") for e in elements],
                ),
            )
        )
    finite = [v for v in times.values() if isinstance(v, (int, float))]
    metrics = dict(
        completed=len(accepted) if acceptance_known else None,
        attempted=len(document.pages),
        median_seconds=statistics.median(finite) if finite else None,
        memory_mib=inference.get("peak_sampled_device_memory_mib"),
        memory_scope="Sampled total GPU-device memory, not allocator peak",
        timing_scope=(
            "Inherited OCR page timings; later pipeline work excluded"
            if spec.stage != "ocr"
            else "Backend-specific page timings; preprocessing/startup scopes differ"
        ),
    )
    return dict(
        pages=pages,
        metrics=metrics,
        document_id=document.document_id,
        canonical=True,
        document="document.json",
        markdown="document.md",
    ), document


def experimental_run(run: Path, target: Path) -> tuple[dict[str, Any], Document]:
    status = json.loads(safe_file(run, "run-status.json").read_text())
    prepared = Path(status["prepared_bundle"]).resolve()
    # The original bundle is metadata, not an arbitrary path supplied by a browser.
    if not prepared.is_relative_to(run.parent.parent.resolve() / "parsed_documents"):
        raise ValueError("Experimental source bundle is outside prepared artifacts")
    document = Document.model_validate_json(safe_file(prepared, "document.json").read_bytes())
    if document.document_id != status["document_id"]:
        raise ValueError("Experimental source identity mismatch")
    target.mkdir(parents=True)
    for asset in (run / "assets").rglob("*"):
        if asset.is_file() and asset.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            copy_asset(safe_file(run, str(asset.relative_to(run))), target / asset.relative_to(run))
    page_status = {p["page_number"]: p for p in status.get("pages", [])}
    pages = []
    for page in document.pages:
        number = page.page_number
        entry = page_status.get(number, {})
        path = run / f"page_{number:04}.json"
        raw = json.loads(path.read_text()) if path.exists() else {}
        markdown_path = run / f"page_{number:04}.md"
        markdown = markdown_path.read_text() if markdown_path.exists() else raw.get("text", "")
        complete = (
            entry.get("status") == "completed_generation" and raw.get("finish_reason") == "stop"
        )
        message = (
            "Generated output; not canonical document JSON. Full-page accuracy is unscored; "
            "selected source checks, where available, are reported separately under Evidence."
            if complete
            else "Incomplete or failed generation. Available text is diagnostic only."
        )
        body = "<aside>" + message + "</aside>" + local_preview(markdown)
        folder = target / "pages"
        folder.mkdir(exist_ok=True)
        (folder / f"{number:04}-preview.html").write_text(page_shell(body))
        pages.append(
            dict(
                number=number,
                status="generated" if complete else "incomplete",
                seconds=entry.get("elapsed_seconds"),
                elements=None,
                content=f"pages/{number:04}-preview.html",
                preview=f"pages/{number:04}-preview.html",
                markdown=markdown,
                raw=raw.get("text"),
                json=dict(
                    page_number=number,
                    finish_reason=raw.get("finish_reason"),
                    text=raw.get("text"),
                    status=entry.get("status", "missing"),
                    canonical_document=False,
                ),
            )
        )
    finite = [p["seconds"] for p in pages if isinstance(p["seconds"], (int, float))]
    memory = [
        p.get("peak_allocated_mib")
        for p in status.get("pages", [])
        if isinstance(p.get("peak_allocated_mib"), (int, float))
    ]
    return dict(
        pages=pages,
        canonical=False,
        document=None,
        markdown=None,
        document_id=document.document_id,
        metrics=dict(
            completed=sum(p["status"] == "generated" for p in pages),
            attempted=len(pages),
            median_seconds=statistics.median(finite) if finite else None,
            memory_mib=max(memory) if memory else None,
            memory_scope="PyTorch peak allocated; not total device memory",
            timing_scope="Generation timing; per-page model loading/startup excluded",
        ),
    ), document


def build_showcase(
    project: Path, destination: Path, manifest_path: Path, *, include_private: bool = False
) -> Path:
    project = project.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Use a fresh showcase directory; saved outputs are never overwritten")
    source_root = project / "outputs/parser_runs"
    if destination.is_relative_to(source_root):
        raise ValueError("Showcase must be outside parser runs")
    manifest = ShowcaseManifest.model_validate_json(manifest_path.read_bytes())
    ids = [r.id for c in manifest.cases for r in c.runs]
    if len(ids) != len(set(ids)) or len({c.id for c in manifest.cases}) != len(manifest.cases):
        raise ValueError("Showcase IDs must be unique")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".showcase-", dir=destination.parent))
    catalog: dict[str, Any] = dict(
        version="1.0",
        private_inputs_included=include_private,
        cases=[],
        runs={},
        limitations=(
            "Completion, agreement and output volume are not accuracy. "
            "Timing/memory scopes differ across backends. Source-reviewed "
            "pipeline output is not raw-model quality."
        ),
    )
    try:
        for case in manifest.cases:
            if case.private and not include_private:
                continue
            data = case.model_dump(exclude={"runs"})
            data["runs"] = []
            document = None
            for spec in case.runs:
                run = source_root / spec.directory
                if not run.resolve().is_relative_to(source_root.resolve()):
                    raise ValueError("Saved run resolves outside parser artifacts")
                if not run.exists():
                    catalog["runs"][spec.id] = dict(
                        id=spec.id, label=spec.label, available=False, stage=spec.stage
                    )
                    data["runs"].append(spec.id)
                    continue
                target = staging / "runs" / spec.id
                payload, current = (
                    canonical_run(run, target, spec)
                    if spec.kind == "canonical"
                    else experimental_run(run, target)
                )
                if document is None:
                    document = current
                    common = staging / "cases" / case.id
                    source = safe_file(run, "source.pdf")
                    if "sha256:" + digest(source) != current.document_id:
                        raise ValueError("Source PDF hash does not match document identity")
                    copy_asset(source, common / "source.pdf")
                    for page in current.pages:
                        source_asset = (
                            (run / page.image.path)
                            if spec.kind == "canonical"
                            else Path(
                                json.loads((run / "run-status.json").read_text())["prepared_bundle"]
                            )
                            / page.image.path
                        )
                        copy_asset(source_asset, common / f"page-{page.page_number:04}.png")
                    data.update(
                        document_id=current.document_id,
                        page_count=len(current.pages),
                        source=f"cases/{case.id}/source.pdf",
                        page_images=[
                            f"cases/{case.id}/page-{p.page_number:04}.png" for p in current.pages
                        ],
                    )
                elif current.document_id != document.document_id or len(current.pages) != len(
                    document.pages
                ):
                    raise ValueError("Compared runs do not share source identity and page count")
                if len(case.original_pages) != len(current.pages):
                    raise ValueError("Original-page map differs from source selection")
                save(target / "run-data.json", payload)
                catalog["runs"][spec.id] = dict(
                    id=spec.id,
                    label=spec.label,
                    stage=spec.stage,
                    available=True,
                    kind=spec.kind,
                    path=f"runs/{spec.id}/run-data.json",
                    base=f"runs/{spec.id}/",
                    metrics=payload["metrics"],
                    canonical=payload["canonical"],
                )
                data["runs"].append(spec.id)
            if document is not None:
                catalog["cases"].append(data)
        if not catalog["cases"]:
            raise ValueError("No saved source cases are available for this showcase")
        for name in [
            "rag-structural-comparison-v2.json",
            "rag-comparison-source-review-v1.json",
            "parser-comparison-showcase-v1.json",
            "gemma-selection-checkpoint-v1.json",
            "author-handwritten-mathematics-comparison-v1.json",
        ]:
            path = project / "evaluation/results" / name
            if path.exists():
                copy_asset(path, staging / "evidence" / name)
                if name == "author-handwritten-mathematics-comparison-v1.json":
                    catalog["math_evaluation"] = "evidence/" + name
        save(staging / "catalog.json", catalog)
        (staging / "index.html").write_text(
            Path(__file__)
            .with_name("showcase_template.html")
            .read_text()
            .replace("__DEMO_VERSION__", "local")
        )
        save(
            staging / "build-manifest.json",
            dict(
                version="1.0",
                inference_requests=0,
                private_inputs_included=include_private,
                case_manifest_sha256=digest(manifest_path),
                artifacts={
                    str(p.relative_to(staging)): digest(p)
                    for p in staging.rglob("*")
                    if p.is_file()
                },
            ),
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "index.html"
