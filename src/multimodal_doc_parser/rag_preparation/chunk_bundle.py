"""Publish validated chunk bundles and a local source/parent/child inspector."""

import hashlib
import json
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.rag_preparation.chunk_builder import build_chunks, validate_chunks
from multimodal_doc_parser.rag_preparation.chunk_schema import (
    ChildRecord,
    ChunkSettings,
    ParentRecord,
)
from multimodal_doc_parser.rag_preparation.token_budget import TokenBudget
from multimodal_doc_parser.visual_enrichment.ollama import safe_asset


def read_records(path: Path, kind: type[ParentRecord] | type[ChildRecord]) -> list[Any]:
    return [kind.model_validate_json(line) for line in path.read_text().splitlines() if line]


def prepare_chunks(source: Path, destination: Path, settings: ChunkSettings | None = None) -> Path:
    settings = settings or ChunkSettings()
    if destination.exists():
        raise ValueError("Use a fresh RAG output directory; existing bundles are never overwritten")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("RAG output must be outside the source parser run")
    source_data = (source / "document.json").read_bytes()
    document = Document.model_validate_json(source_data)
    budget = TokenBudget()
    parents, children, excluded = build_chunks(document, settings, budget)
    validation = validate_chunks(document, parents, children, excluded, settings, budget)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".chunking-", dir=destination.parent))
    try:
        for name in ["parents", "children"]:
            records = parents if name == "parents" else children
            (staging / f"{name}.jsonl").write_text(
                "".join(r.model_dump_json() + "\n" for r in records)
            )
        (staging / "source_document.json").write_bytes(source_data)
        if (source / "source.pdf").exists():
            shutil.copyfile(source / "source.pdf", staging / "source.pdf")
        for element in document.elements:
            asset = getattr(element, "asset", None)
            if asset is not None:
                path = safe_asset(source, asset.path)
                target = safe_asset(staging, asset.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        manifest: dict[str, Any] = dict(
            schema_version="1.0",
            document_id=document.document_id,
            source_document_sha256=hashlib.sha256(source_data).hexdigest(),
            settings=settings.model_dump(),
            tokenizer_package_version=version("tiktoken"),
            token_budget_note=(
                "cl100k_base counts are explicit generation budgets; "
                "BGE windows are separately bounded by its own tokenizer."
            ),
            producer="multimodal-doc-parser",
            algorithm="structure-adaptive-parent-child-v1",
            validation=validation,
            source_run=str(source),
            inference=dict(ocr_pages=0, vlm_requests=0, embedding_requests=0),
            artifacts={
                name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                for name in ["parents.jsonl", "children.jsonl", "source_document.json"]
            },
        )
        if (staging / "source.pdf").exists():
            manifest["artifacts"]["source.pdf"] = hashlib.sha256(
                (staging / "source.pdf").read_bytes()
            ).hexdigest()
        (staging / "chunk_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        write_preview(staging, parents, children, manifest)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def write_preview(
    destination: Path,
    parents: list[ParentRecord],
    children: list[ChildRecord],
    manifest: dict[str, Any],
) -> None:
    payload = (
        json.dumps(
            dict(
                parents=[p.model_dump(mode="json") for p in parents],
                children=[c.model_dump(mode="json") for c in children],
                manifest=manifest,
            ),
            ensure_ascii=False,
        )
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    template = Path(__file__).with_name("chunk_preview.html").read_text()
    (destination / "review.html").write_text(template.replace("__CHUNK_DATA__", payload))
