"""Source-addressable evidence and immutable parent/child chunk contracts."""

from typing import Literal, Self

from pydantic import Field, model_validator

from multimodal_doc_parser.document_schema.document import (
    DocumentElement,
    Provenance,
    SchemaModel,
    SourceRegion,
)


class ChunkSettings(SchemaModel):
    tokenizer: Literal["cl100k_base"] = "cl100k_base"
    child_target_tokens: int = Field(default=350, ge=64, le=1000)
    child_max_tokens: int = Field(default=500, ge=128, le=2000)
    parent_max_tokens: int = Field(default=1800, ge=256, le=8000)

    @model_validator(mode="after")
    def coherent_limits(self) -> Self:
        if not self.child_target_tokens <= self.child_max_tokens <= self.parent_max_tokens:
            raise ValueError("Require child target <= child maximum <= parent maximum")
        return self


class CellReference(SchemaModel):
    element_id: str
    row: int = Field(ge=0)
    column: int = Field(ge=0)


class EvidencePart(SchemaModel):
    id: str
    role: Literal[
        "heading",
        "text",
        "formula",
        "caption",
        "footnote",
        "table_header",
        "table_row",
        "description",
        "chart_series",
        "visual_analysis",
        "visual_placeholder",
    ]
    text: str
    source_element_ids: list[str] = Field(min_length=1)
    cells: list[CellReference] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    inferred: bool = False
    context_only: bool = False
    column_group: str | None = None


class PartSlice(SchemaModel):
    part_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class ParentRecord(SchemaModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str
    document_id: str
    section_id: str | None
    heading_path: list[str]
    kind: Literal["text", "table", "visual"]
    logical_table_id: str | None = None
    source_elements: list[DocumentElement]
    parts: list[EvidencePart]
    child_ids: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)


class ChildRecord(SchemaModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str
    parent_id: str
    document_id: str
    heading_path: list[str]
    kind: Literal["text", "table", "visual"]
    text: str
    token_count: int = Field(gt=0)
    slices: list[PartSlice] = Field(min_length=1)
    context_part_ids: list[str]
    deferred_context_part_ids: list[str]
    source_element_ids: list[str]
    source_regions: list[SourceRegion]
    column_group: str | None = None
    previous_child_id: str | None = None
    next_child_id: str | None = None
    quality_warnings: list[str] = Field(default_factory=list)
