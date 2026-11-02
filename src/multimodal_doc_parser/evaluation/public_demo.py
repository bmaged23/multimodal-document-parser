"""Publish allowlisted excerpts from a verified saved showcase without inference."""

import copy
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from multimodal_doc_parser.evaluation.rag_benchmark import digest, save
from multimodal_doc_parser.evaluation.showcase import copy_asset, safe_file


class PublicCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    saved_pages: list[int] = Field(min_length=1)
    attribution: str = Field(min_length=1)
    rights_note: str = Field(min_length=1)


class PublicSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    cases: list[PublicCase] = Field(min_length=1)


def referenced_assets(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(referenced_assets(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(referenced_assets(v) for v in value))
    if isinstance(value, str):
        paths = {value} if value.startswith("assets/") and "\n" not in value else set()
        paths.update(re.findall(r"\]\((assets/[^\s)]+)\)", value))
        paths.update(re.findall(r"""src=["'](assets/[^"']+)["']""", value))
        return paths
    return set()


def build_public_demo(
    showcase: Path,
    destination: Path,
    selection_path: Path,
    *,
    evidence_directory: Path | None = None,
) -> Path:
    showcase, destination = showcase.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(showcase):
        raise ValueError("Use a fresh demo directory outside the saved showcase")
    selection = PublicSelection.model_validate_json(selection_path.read_bytes())
    original = json.loads(safe_file(showcase, "catalog.json").read_text())
    inventory = json.loads(safe_file(showcase, "build-manifest.json").read_text())["artifacts"]
    if original.get("private_inputs_included") is not False:
        raise ValueError("Public demo requires a showcase with private inputs excluded")
    if len({c.id for c in selection.cases}) != len(selection.cases):
        raise ValueError("Duplicate public demo case")

    def verified(relative: str) -> Path:
        path = safe_file(showcase, relative)
        if inventory.get(relative) != digest(path):
            raise ValueError("Saved showcase checksum mismatch: " + relative)
        return path

    verified("catalog.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".public-demo-", dir=destination.parent))
    catalog = {
        **original,
        "version": selection.version,
        "cases": [],
        "runs": {},
        "public_excerpt": True,
    }
    sources = []
    try:
        for spec in selection.cases:
            case = next((c for c in original["cases"] if c["id"] == spec.id), None)
            if case is None or case.get("private") is not False:
                raise ValueError("Unknown or private public demo case: " + spec.id)
            numbers = spec.saved_pages
            if len(set(numbers)) != len(numbers) or any(
                n < 1 or n > case["page_count"] for n in numbers
            ):
                raise ValueError("Invalid or duplicate public page selection")
            current = copy.deepcopy(case)
            current.update(
                original_pages=[case["original_pages"][n - 1] for n in numbers],
                page_count=len(numbers),
                preview_page=1,
                page_images=[],
                description=(
                    f"{len(numbers)} curated demo pages from the {case['page_count']}-page "
                    "saved selection. Metrics describe that full saved run. " + spec.attribution
                ),
            )
            source = verified(case["source"])
            if "sha256:" + digest(source) != case["document_id"]:
                raise ValueError("Saved source identity mismatch")
            target_pdf = staging / case["source"]
            target_pdf.parent.mkdir(parents=True, exist_ok=True)
            with pdfium.PdfDocument(source) as pdf, pdfium.PdfDocument.new() as excerpt:
                excerpt.import_pages(pdf, [n - 1 for n in numbers])
                excerpt.save(target_pdf)
            current["excerpt_pdf_sha256"] = digest(target_pdf)
            for public_number, saved_number in enumerate(numbers, start=1):
                relative = f"cases/{spec.id}/page-{public_number:04}.png"
                copy_asset(verified(case["page_images"][saved_number - 1]), staging / relative)
                current["page_images"].append(relative)

            for run_id in case["runs"]:
                meta = copy.deepcopy(original["runs"][run_id])
                catalog["runs"][run_id] = meta
                if not meta["available"]:
                    continue
                payload = json.loads(verified(meta["path"]).read_text())
                pages = []
                markdown = []
                for public_number, saved_number in enumerate(numbers, start=1):
                    page = copy.deepcopy(
                        next(p for p in payload["pages"] if p["number"] == saved_number)
                    )
                    page.update(number=public_number, saved_page_number=saved_number)
                    for key in ["content", "preview"]:
                        relative = meta["base"] + page[key]
                        html = verified(relative).read_text()
                        target_html = staging / relative
                        target_html.parent.mkdir(parents=True, exist_ok=True)
                        target_html.write_text(
                            html.replace(
                                "</style>",
                                "td{white-space:nowrap}th{overflow-wrap:normal}"
                                ".table-scroll{overflow:auto}</style>",
                            )
                        )
                        for asset in referenced_assets(html):
                            relative_asset = meta["base"] + asset
                            copy_asset(verified(relative_asset), staging / relative_asset)
                    for asset in referenced_assets(page):
                        relative_asset = meta["base"] + asset
                        copy_asset(verified(relative_asset), staging / relative_asset)
                    pages.append(page)
                    markdown.append(
                        f"<!-- original PDF page {case['original_pages'][saved_number - 1]}; "
                        f"saved selection page {saved_number} -->\n\n" + page["markdown"]
                    )
                payload["pages"] = pages
                payload["public_excerpt"] = True
                payload["document"] = "page-excerpts.json" if payload["canonical"] else None
                payload["markdown"] = "markdown-excerpts.md"
                payload["metrics"]["timing_scope"] += "; full saved selection, not demo subset"
                payload["metrics"]["memory_scope"] += "; full saved run, not remeasured"
                save(staging / meta["path"], payload)
                save(
                    staging / meta["base"] / "page-excerpts.json",
                    dict(
                        source_document_id=case["document_id"],
                        note="Per-page saved excerpts; not a complete canonical Document export.",
                        original_pdf_pages=current["original_pages"],
                        pages=[p["json"] for p in pages],
                    ),
                )
                (staging / meta["base"] / "markdown-excerpts.md").write_text("\n\n".join(markdown))
            catalog["cases"].append(current)
            sources.append(
                dict(
                    case=spec.id,
                    saved_pages=numbers,
                    original_pdf_pages=current["original_pages"],
                    original_document_id=case["document_id"],
                    excerpt_pdf_sha256=current["excerpt_pdf_sha256"],
                    attribution=spec.attribution,
                    rights_note=spec.rights_note,
                    source_url=case["source_url"],
                )
            )
        for relative in inventory:
            if relative.startswith("evidence/"):
                if evidence_directory is None:
                    copy_asset(verified(relative), staging / relative)
                else:
                    copy_asset(
                        safe_file(evidence_directory, Path(relative).name), staging / relative
                    )
        verified("index.html")
        viewer_template = Path(__file__).with_name("showcase_template.html")
        template = viewer_template.read_text().replace("__DEMO_VERSION__", selection.version)
        template = template.replace(
            "Completion is not accuracy", "Full saved selection; not just demo excerpts"
        ).replace("Read-only artifact showcase", "Public saved-output demo · no live inference")
        (staging / "index.html").write_text(template)
        (staging / ".nojekyll").touch()
        save(staging / "catalog.json", catalog)
        save(
            staging / "build-manifest.json",
            dict(
                version=selection.version,
                inference_requests=0,
                private_inputs_included=False,
                selection_sha256=digest(selection_path),
                source_catalog_sha256=digest(showcase / "catalog.json"),
                viewer_template_sha256=digest(viewer_template),
                presentation="Numeric table tokens stay intact; source and predictions unchanged.",
                sources=sources,
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
