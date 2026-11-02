"""Portable BM25/vector indexes and token-bounded parent expansion."""

import hashlib
import importlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from multimodal_doc_parser.rag_preparation.chunk_builder import context_for, render_slices
from multimodal_doc_parser.rag_preparation.chunk_bundle import read_records
from multimodal_doc_parser.rag_preparation.chunk_schema import ChildRecord, ParentRecord, PartSlice
from multimodal_doc_parser.rag_preparation.gpu_embeddings import DIMENSIONS, GPUEmbeddings
from multimodal_doc_parser.rag_preparation.token_budget import TokenBudget


def verify_bundle(bundle: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((bundle / "chunk_manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError("Unsafe artifact manifest path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("RAG bundle artifact checksum mismatch: " + name)
    return manifest


def build_index(
    bundles: list[Path], destination: Path, *, embedder: Any = None, batch_size: int = 2
) -> Path:
    if destination.exists():
        raise ValueError("Use a fresh index directory")
    if not bundles or not 1 <= batch_size <= 8:
        raise ValueError("Provide bundles and a GPU batch size between 1 and 8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".indexing-", dir=destination.parent))
    try:
        parents: list[ParentRecord] = []
        children: list[ChildRecord] = []
        sources = []
        for bundle in bundles:
            manifest = verify_bundle(bundle)
            parents.extend(read_records(bundle / "parents.jsonl", ParentRecord))
            children.extend(read_records(bundle / "children.jsonl", ChildRecord))
            sources.append(
                dict(
                    document_id=manifest["document_id"],
                    bundle=os.path.relpath(bundle.resolve(), destination.resolve()),
                    manifest_sha256=hashlib.sha256(
                        (bundle / "chunk_manifest.json").read_bytes()
                    ).hexdigest(),
                )
            )
        if not children or len({c.id for c in children}) != len(children):
            raise ValueError("Index children must be nonempty and unique across bundles")
        if len({p.document_id for p in parents}) != len(bundles):
            raise ValueError("Index each source document only once")
        (staging / "parents.jsonl").write_text("".join(p.model_dump_json() + "\n" for p in parents))
        (staging / "children.jsonl").write_text(
            "".join(c.model_dump_json() + "\n" for c in children)
        )
        with sqlite3.connect(staging / "keyword.sqlite") as connection:
            connection.execute(
                "CREATE VIRTUAL TABLE children USING fts5(child_id UNINDEXED, text, "
                "tokenize='porter unicode61')"
            )
            connection.executemany(
                "INSERT INTO children VALUES (?, ?)", [(c.id, c.text) for c in children]
            )
        embedding_metadata = None
        windows = []
        if embedder is not None:
            np = importlib.import_module("numpy")

            texts = []
            for child in children:
                for start, end in embedder.windows(child.text):
                    texts.append(child.text[start:end])
                    windows.append(dict(child_id=child.id, start=start, end=end))
            vectors = []
            for start in range(0, len(texts), batch_size):
                vectors.append(embedder.encode(texts[start : start + batch_size]))
                if start % 64 == 0 or start + batch_size >= len(texts):
                    print(
                        f"Embedded {min(start + batch_size, len(texts))}/{len(texts)} windows",
                        flush=True,
                    )
            matrix = np.concatenate(vectors, axis=0)
            if matrix.shape != (len(windows), DIMENSIONS):
                raise ValueError("Index requires exactly 1024 embedding dimensions")
            np.save(staging / "vectors.npy", matrix, allow_pickle=False)
            (staging / "embedding_windows.json").write_text(json.dumps(windows) + "\n")
            embedding_metadata = embedder.metadata()
        manifest = dict(
            schema_version="1.0",
            children=len(children),
            parents=len(parents),
            source_bundles=sources,
            keyword_search="SQLite FTS5 BM25",
            fusion="reciprocal-rank-fusion k=60; equal weights",
            embedding=embedding_metadata,
            embedding_windows=len(windows),
            batch_size=batch_size,
            artifacts={
                name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                for name in ["parents.jsonl", "children.jsonl", "keyword.sqlite"]
                + (["vectors.npy", "embedding_windows.json"] if embedding_metadata else [])
            },
        )
        (staging / "index_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


class Retriever:
    def __init__(self, directory: Path, embedder: Any = None) -> None:
        self.directory = directory
        self.manifest = json.loads((directory / "index_manifest.json").read_text())
        for name, expected in self.manifest["artifacts"].items():
            if (
                Path(name).name != name
                or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected
            ):
                raise ValueError("Index artifact checksum mismatch")
        self.children: list[ChildRecord] = read_records(directory / "children.jsonl", ChildRecord)
        self.by_id = {c.id: c for c in self.children}
        self.parents = {p.id: p for p in read_records(directory / "parents.jsonl", ParentRecord)}
        self.embedder = embedder
        self.vectors: Any = None
        self.windows: list[dict[str, Any]] = []
        self.budget = TokenBudget()
        if self.manifest["embedding"]:
            np = importlib.import_module("numpy")

            self.vectors = np.load(directory / "vectors.npy", allow_pickle=False)
            self.windows = json.loads((directory / "embedding_windows.json").read_text())
            if (
                self.vectors.shape != (len(self.windows), DIMENSIONS)
                or not np.isfinite(self.vectors).all()
            ):
                raise ValueError("Expected finite 1024-dimensional index vectors")
            if any(w["child_id"] not in self.by_id for w in self.windows):
                raise ValueError("Embedding window references unknown child")

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        context_tokens: int = 2400,
        expand: bool = True,
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        if not query.strip() or not 1 <= top_k <= 30 or not 128 <= context_tokens <= 12000:
            raise ValueError("Provide a query, top_k 1–30, context_tokens 128–12000")
        if mode not in {"keyword", "hybrid", "dense"}:
            raise ValueError("Unknown retrieval mode")
        ranks: dict[str, float] = defaultdict(float)
        details: dict[str, dict[str, Any]] = defaultdict(dict)
        candidates = max(top_k * 4, 20)
        terms = re.findall(r"\w+", query, re.UNICODE)
        expression = " OR ".join('"' + t + '"' for t in dict.fromkeys(terms))
        if expression and mode != "dense":
            with sqlite3.connect(
                f"file:{self.directory / 'keyword.sqlite'}?mode=ro", uri=True
            ) as connection:
                found = connection.execute(
                    (
                        "SELECT child_id, bm25(children) FROM children WHERE children MATCH ? "
                        "ORDER BY bm25(children), child_id LIMIT ?"
                    ),
                    (expression, candidates),
                ).fetchall()
            for rank, (child_id, value) in enumerate(found, 1):
                ranks[child_id] += 1 / (60 + rank)
                details[child_id].update(keyword_rank=rank, bm25=value)
        if mode != "keyword":
            if self.vectors is None:
                raise ValueError("Dense/hybrid retrieval requires a GPU embedding index")
            metadata = self.manifest["embedding"]
            if self.embedder is None:
                self.embedder = GPUEmbeddings(revision=metadata["revision"])
            actual = self.embedder.metadata()
            if any(
                actual.get(k) != metadata.get(k)
                for k in ["model", "revision", "dimensions", "pooling", "normalization"]
            ):
                raise ValueError("Query embedding model is incompatible with index")
            vector = self.embedder.query(query)
            scores = self.vectors @ vector
            ranked: dict[str, float] = {}
            for window, value in zip(self.windows, scores, strict=True):
                child_id = window["child_id"]
                ranked[child_id] = max(float(value), ranked.get(child_id, -1.0))
            for rank, (child_id, value) in enumerate(
                sorted(ranked.items(), key=lambda p: (-p[1], p[0]))[:candidates], 1
            ):
                ranks[child_id] += 1 / (60 + rank)
                details[child_id].update(dense_rank=rank, cosine_similarity=value)
        matched = sorted(ranks, key=lambda key: (-ranks[key], key))[:top_k]
        hits = [
            dict(
                child_id=key, parent_id=self.by_id[key].parent_id, score=ranks[key], **details[key]
            )
            for key in matched
        ]
        contexts = self.expand(matched, context_tokens, expand)
        return dict(
            query=query,
            mode=mode,
            matched_children=hits,
            contexts=contexts,
            context_tokens=self.budget.count("\n\n".join(c["text"] for c in contexts)),
            context_token_limit=context_tokens,
            expanded=expand,
            citation_note=(
                "Evidence IDs identify source locations; they are not independent "
                "correctness verification."
            ),
        )

    def expand(self, matched: list[str], maximum: int, expand: bool) -> list[dict[str, Any]]:
        groups: dict[str, list[ChildRecord]] = {}
        for identity in matched:
            child = self.by_id[identity]
            # Without expansion retain each child as a separate context block.
            key = child.parent_id if expand else child.id
            groups.setdefault(key, []).append(child)
        output: list[dict[str, Any]] = []
        for children in groups.values():
            parent = self.parents[children[0].parent_id]
            parts = {p.id: p for p in parent.parts}
            chosen: list[PartSlice] = []
            accepted = []
            prefix = f"Evidence parent {parent.id}; document {parent.document_id}\n"

            def fits(
                slices: list[PartSlice],
                contexts: list[str],
                parent: ParentRecord = parent,
                prefix: str = prefix,
            ) -> bool:
                text = prefix + render_slices(parent, slices, contexts)
                return (
                    self.budget.count("\n\n".join([c["text"] for c in output] + [text])) <= maximum
                )

            for child in children:
                additions = [s for s in child.slices if s not in chosen]
                if fits(chosen + additions, []):
                    chosen.extend(additions)
                    accepted.append(child.id)
            if not chosen:
                continue
            context: list[str] = []
            available = context_for(parent, chosen)
            for item in sorted(
                available,
                key=lambda p: (
                    {"table_header": 0, "caption": 1, "footnote": 2, "heading": 3}.get(p.role, 4),
                    p.id,
                ),
            ):
                if fits(chosen, context + [item.id]):
                    context.append(item.id)
            if expand and (
                parent.kind != "table"
                or sum(e.type == "table" for e in parent.source_elements) == 1
            ):
                entire = [
                    PartSlice(part_id=p.id, start=0, end=len(p.text))
                    for p in parent.parts
                    if not p.context_only and p.text
                ]
                entire.extend(s for s in chosen if parts[s.part_id].context_only)
                if (
                    fits(entire, context)
                    and self.budget.count(render_slices(parent, entire, context)) <= 1800
                ):
                    chosen = entire
            # Preserve source order rather than retrieval rank inside a parent.
            order = {p.id: i for i, p in enumerate(parent.parts)}
            chosen.sort(key=lambda s: (order[s.part_id], s.start))
            source_ids = list(
                dict.fromkeys(
                    i
                    for p in context + [s.part_id for s in chosen]
                    for i in parts[p].source_element_ids
                )
            )
            elements = {e.id: e for e in parent.source_elements}
            regions = list(
                {
                    r.model_dump_json(): r.model_dump()
                    for i in source_ids
                    for r in elements[i].source_regions
                }.values()
            )
            omitted = [p.id for p in available if p.id not in context]
            output.append(
                dict(
                    parent_id=parent.id,
                    document_id=parent.document_id,
                    matched_child_ids=accepted,
                    source_element_ids=source_ids,
                    source_regions=regions,
                    evidence_slices=[s.model_dump() for s in chosen],
                    context_part_ids=context,
                    deferred_context_part_ids=omitted,
                    quality_warnings=parent.quality_warnings
                    + (
                        ["Some parent context did not fit the query context budget."]
                        if omitted
                        else []
                    ),
                    text=prefix + render_slices(parent, chosen, context),
                )
            )
        return output
