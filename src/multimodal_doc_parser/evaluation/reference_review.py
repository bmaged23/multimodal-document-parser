"""Local, source-only reference annotation review with immutable drafts and revision history."""

import copy
import hashlib
import json
import math
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from multimodal_doc_parser.document_schema.document import BoundingBox, SchemaModel

Status = Literal["draft", "approved", "ambiguous", "excluded"]
KINDS = {
    "text",
    "formula",
    "table_cells",
    "chart_values",
    "layout_region",
    "headings",
    "reading_order",
    "page_inventory",
    "diagram_edges",
    "row_values",
}


class RevisionConflict(ValueError):
    """The browser edited an outdated annotation revision."""


class ReviewUpdate(SchemaModel):
    expected_revision: int = Field(ge=0, strict=True)
    status: Status
    reviewer: str = Field(max_length=100)
    value: dict[str, Any]
    source_box: BoundingBox | None
    note: str = Field(max_length=5000)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_atomic(path: Path, value: Any) -> None:
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as f:
        f.write(serialized)
        f.flush()
        os.fsync(f.fileno())
        temporary = Path(f.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_value(kind: str, value: dict[str, Any]) -> None:
    if kind not in KINDS:
        raise ValueError("Unknown annotation kind")
    if kind in {"text", "formula"}:
        if not isinstance(value.get("text"), str) or not value["text"].strip():
            raise ValueError("A transcription is required")
        if value.get("scope") not in {"full_page", "selected_span"}:
            raise ValueError("Declare full_page or selected_span scope")
    elif kind == "table_cells":
        cells = value.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError("At least one reviewed table cell is required")
        seen = set()
        for c in cells:
            for key in ["row", "column"]:
                if type(c.get(key)) is not int or c[key] < 0:
                    raise ValueError("Cell row/column must be nonnegative integers")
            cell_origin = (c["row"], c["column"])
            if cell_origin in seen:
                raise ValueError("Duplicate cell origin")
            seen.add(cell_origin)
            if not isinstance(c.get("text"), str):
                raise ValueError("Every cell needs text, including empty cells")
            for attr in ["row_span", "column_span"]:
                if c.get(attr) is not None and (type(c[attr]) is not int or c[attr] < 1):
                    raise ValueError("Cell spans must be positive integers or unannotated/null")
            if c.get("is_header") is not None and type(c["is_header"]) is not bool:
                raise ValueError("Header flag must be boolean or unannotated/null")
        if value.get("scope") != "selected_cells":
            raise ValueError("These sparse drafts annotate selected_cells, not a complete grid")
    elif kind == "layout_region":
        if value.get("label") not in {
            "text",
            "heading",
            "table",
            "chart",
            "image",
            "diagram",
            "decoration",
        }:
            raise ValueError("Choose a supported region label")
    elif kind == "reading_order":
        anchors = value.get("anchors")
        if (
            not isinstance(anchors, list)
            or len(anchors) < 2
            or not all(isinstance(a, str) and a.strip() for a in anchors)
        ):
            raise ValueError("Reading order needs at least two nonempty source anchors")
    elif kind == "headings":
        headings = value.get("headings")
        if not isinstance(headings, list) or not headings:
            raise ValueError("Add at least one heading")
        ids = [h.get("id") for h in headings]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Heading IDs must be unique nonempty strings")
        parents = {h["id"]: h.get("parent_id") for h in headings}
        for h in headings:
            if not isinstance(h.get("text"), str) or not h["text"].strip():
                raise ValueError("Heading text is required")
            if type(h.get("level")) is not int or not 1 <= h["level"] <= 6:
                raise ValueError("Heading level must be 1–6")
            current, visited = h["id"], set()
            while current is not None:
                if current in visited or current not in parents:
                    raise ValueError("Heading ancestry contains a cycle or missing parent")
                visited.add(current)
                current = parents[current]
    elif kind == "chart_values":
        points = value.get("printed_points")
        if not isinstance(points, list) or not points:
            raise ValueError("At least one explicitly printed chart point is required")
        seen = set()
        for p in points:
            if not all(isinstance(p.get(k), str) and p[k].strip() for k in ["series", "label"]):
                raise ValueError("Printed points require series and category/year labels")
            number = p.get("value")
            if type(number) not in {int, float} or not math.isfinite(number):
                raise ValueError("Printed chart values must be finite numbers")
            point_key = (p["series"], p["label"])
            if point_key in seen:
                raise ValueError("Duplicate chart series/category")
            seen.add(point_key)
    elif kind == "diagram_edges":
        edges = value.get("edges")
        if (
            not isinstance(edges, list)
            or not edges
            or not all(
                isinstance(e.get("source"), str)
                and e["source"].strip()
                and isinstance(e.get("target"), str)
                and e["target"].strip()
                for e in edges
            )
        ):
            raise ValueError("Directed edges require source and target labels")
    elif kind == "row_values":
        rows = value.get("rows")
        if (
            not isinstance(rows, list)
            or not rows
            or not all(
                isinstance(r.get("label"), str) and isinstance(r.get("values"), list) for r in rows
            )
        ):
            raise ValueError("Row-value checks require labeled value lists")
    elif kind == "page_inventory":
        for key in ["table_count", "substantive_visual_count"]:
            if type(value.get(key)) is not int or value[key] < 0:
                raise ValueError("Inventory counts must be nonnegative integers")


class ReviewStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.dataset_path = self.directory / "dataset.json"
        self.dataset: dict[str, Any] = json.loads(self.dataset_path.read_text())
        self.fingerprint = digest(self.dataset_path)
        self.state_path = self.directory / "review-state.json"
        self.lock = threading.RLock()
        self.pages = {(p["case"], p["page"]): p for p in self.dataset["pages"]}
        self.initial = {i["id"]: i for i in self.dataset["annotations"]}
        self.state: dict[str, Any] = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else dict(
                dataset_sha256=self.fingerprint,
                annotations={
                    i: copy.deepcopy(a)
                    | dict(
                        status="draft",
                        reviewer="",
                        note=a.get("note", ""),
                        revision=0,
                        history=[],
                    )
                    for i, a in self.initial.items()
                },
            )
        )
        if self.state["dataset_sha256"] != self.fingerprint:
            raise ValueError("Draft dataset changed; use a new review workspace")

    def source(self, case: str, page: int) -> Path:
        record = self.pages[(case, page)]
        path: Path = (self.directory / record["image"]).resolve()
        if not path.is_relative_to(self.directory) or digest(path) != record["image_sha256"]:
            raise ValueError("Source image is unsafe or differs from the frozen draft")
        return path

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            result: dict[str, Any] = copy.deepcopy(self.dataset)
            result["annotations"] = [
                {**a, "initial_value": self.initial[a["id"]]["value"]}
                for a in self.state["annotations"].values()
            ]
            result["dataset_sha256"] = self.fingerprint
            return result

    def update(self, identity: str, update: ReviewUpdate) -> dict[str, Any]:
        with self.lock:
            old = self.state["annotations"][identity]
            if old["revision"] != update.expected_revision:
                raise RevisionConflict("Annotation changed in another tab; reload before saving")
            if update.status != "draft" and not update.reviewer.strip():
                raise ValueError("Enter a reviewer name before recording a review decision")
            if update.status in {"ambiguous", "excluded"} and not update.note.strip():
                raise ValueError("Record why this annotation is ambiguous or excluded")
            self.source(old["case"], old["page"])
            if update.status == "approved":
                validate_value(old["kind"], update.value)
                if (
                    old["kind"] not in {"page_inventory", "reading_order"}
                    and update.source_box is None
                ):
                    raise ValueError("Draw a source region before approving this annotation")
            candidate = copy.deepcopy(self.state)
            event = dict(at_utc=datetime.now(UTC).isoformat(), **update.model_dump())
            edited = dict(
                **{
                    k: v
                    for k, v in old.items()
                    if k
                    not in {
                        "status",
                        "reviewer",
                        "note",
                        "value",
                        "source_box",
                        "revision",
                        "history",
                    }
                },
                status=update.status,
                reviewer=update.reviewer.strip(),
                note=update.note,
                value=update.value,
                source_box=update.source_box.model_dump() if update.source_box else None,
                revision=old["revision"] + 1,
                history=[*old["history"], event],
            )
            candidate["annotations"][identity] = edited
            save_atomic(self.state_path, candidate)
            self.state = candidate
            return {**copy.deepcopy(edited), "initial_value": self.initial[identity]["value"]}

    def export(self) -> dict[str, Any]:
        with self.lock:
            approved = []
            for a in self.state["annotations"].values():
                if a["status"] != "approved":
                    continue
                self.source(a["case"], a["page"])
                validate_value(a["kind"], a["value"])
                approved.append(copy.deepcopy(a))
            counts = {
                s: sum(a["status"] == s for a in self.state["annotations"].values())
                for s in ["draft", "approved", "ambiguous", "excluded"]
            }
            exported = dict(
                version="reviewed-source-reference-v1",
                dataset_sha256=self.fingerprint,
                annotation_status=(
                    "User-reviewed assistant/source drafts; not independently annotated or blind"
                ),
                reviewer_identity=(
                    "Self-reported local reviewer names; not authenticated identities"
                ),
                source_pages=self.dataset["pages"],
                counts=counts,
                annotations=approved,
                scope=(
                    "Only explicitly approved regions/assertions. "
                    "Do not infer full-page accuracy from selected spans/cells."
                ),
                created_at_utc=datetime.now(UTC).isoformat(),
            )
            target = self.directory / "exports"
            target.mkdir(exist_ok=True)
            filename = datetime.now(UTC).strftime("reviewed-references-%Y%m%dT%H%M%S%fZ.json")
            save_atomic(target / filename, exported)
            return exported
