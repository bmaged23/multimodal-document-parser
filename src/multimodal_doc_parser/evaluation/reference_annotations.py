"""Explicit, source-bound reference annotations for limited parser evaluation."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TableCellReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_number: int = Field(strict=True, ge=1)
    table_index: int = Field(strict=True, ge=0)
    row: int = Field(strict=True, ge=0)
    column: int = Field(strict=True, ge=0)
    text: str
    row_span: int | None = Field(default=None, strict=True, ge=1)
    column_span: int | None = Field(default=None, strict=True, ge=1)


class ReferenceAnnotations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    annotation_provenance: str = Field(min_length=1)
    page_text: dict[int, str] = Field(default_factory=dict)
    table_cells: list[TableCellReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if not self.page_text and not self.table_cells:
            raise ValueError("Reference annotations must include page text or table cells")
        if any(number < 1 or not text.strip() for number, text in self.page_text.items()):
            raise ValueError(
                "Reference transcriptions require positive page numbers and nonempty text"
            )
        identities = [(c.page_number, c.table_index, c.row, c.column) for c in self.table_cells]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate table-cell reference coordinates")
        return self
