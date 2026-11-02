"""Versioned, parser-independent document representation."""

from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Identifier = Annotated[str, Field(min_length=1)]


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BoundingBox(SchemaModel):
    """Normalized top-left coordinates relative to the rendered, rotated page."""

    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered_edges(self) -> Self:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("Bounding box must have positive width and height")
        return self


class SourceRegion(SchemaModel):
    page_number: PositiveInt
    bounding_box: BoundingBox


class AssetReference(SchemaModel):
    path: str

    @model_validator(mode="after")
    def relative_path(self) -> Self:
        path = PurePosixPath(self.path)
        if not self.path or path.is_absolute() or ".." in path.parts or "\\" in self.path:
            raise ValueError("Asset path must be a safe relative POSIX path")
        if ":" in self.path or self.path == ".":
            raise ValueError("Invalid asset path")
        return self


class Page(SchemaModel):
    page_number: PositiveInt
    width_points: float = Field(gt=0)
    height_points: float = Field(gt=0)
    width_pixels: PositiveInt
    height_pixels: PositiveInt
    rotation_degrees: Literal[0, 90, 180, 270]
    image: AssetReference


class Provenance(SchemaModel):
    producer: Identifier
    model_version: str | None = None
    method: Literal["extracted", "generated", "manual"]


class ElementBase(SchemaModel):
    id: Identifier
    parent_id: Identifier | None = None
    source_regions: list[SourceRegion] = Field(min_length=1)
    provenance: Provenance


class TextElement(ElementBase):
    type: Literal["text", "list_item", "caption", "footnote", "page_header", "page_footer"]
    text: str


class HeadingElement(ElementBase):
    type: Literal["heading"] = "heading"
    text: str
    level: int = Field(ge=1, le=6)


class FormulaElement(ElementBase):
    type: Literal["formula"] = "formula"
    latex: str


class TableCell(SchemaModel):
    row: NonNegativeInt
    column: NonNegativeInt
    row_span: PositiveInt = 1
    column_span: PositiveInt = 1
    text: str
    is_header: bool = False
    header_inferred: bool = False


class TableElement(ElementBase):
    type: Literal["table"] = "table"
    caption: str | None = None
    row_count: PositiveInt
    column_count: PositiveInt
    cells: list[TableCell]
    header_inference: str | None = None
    caption_element_id: str | None = None
    footnote_element_ids: list[str] = Field(default_factory=list)
    logical_table_id: str | None = None
    continuation_index: PositiveInt | None = None
    column_group: str | None = None

    @model_validator(mode="after")
    def valid_grid(self) -> Self:
        occupied: set[tuple[int, int]] = set()
        for cell in self.cells:
            if (
                cell.row + cell.row_span > self.row_count
                or cell.column + cell.column_span > self.column_count
            ):
                raise ValueError("Table cell extends beyond the table grid")
            for row in range(cell.row, cell.row + cell.row_span):
                for column in range(cell.column, cell.column + cell.column_span):
                    if (row, column) in occupied:
                        raise ValueError("Table cells overlap")
                    occupied.add((row, column))
        return self


class ChartPoint(SchemaModel):
    label: str
    value: float | None
    value_basis: Literal["printed_label", "visual_estimate", "unknown"]

    @model_validator(mode="after")
    def known_value(self) -> Self:
        if (self.value is None) != (self.value_basis == "unknown"):
            raise ValueError("Unknown values must be null; other values must be numeric")
        return self


class ChartSeries(SchemaModel):
    name: str
    unit: str | None = None
    points: list[ChartPoint]


class FlowNode(SchemaModel):
    id: Identifier
    label: str


class FlowEdge(SchemaModel):
    source: Identifier
    target: Identifier
    label: str | None = None


class VisualAnalysis(SchemaModel):
    kind: Literal["chart", "image", "flowchart", "unknown"]
    description: str = Field(min_length=1, max_length=6000)
    title: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    series: list[ChartSeries] = Field(default_factory=list, max_length=20)
    nodes: list[FlowNode] = Field(default_factory=list, max_length=100)
    edges: list[FlowEdge] = Field(default_factory=list, max_length=150)
    uncertainties: list[str] = Field(default_factory=list, max_length=30)
    plot_type: Literal["bar", "line", "scatter", "composite", "other"] | None = None

    @model_validator(mode="after")
    def coherent_visual(self) -> Self:
        if self.kind != "chart" and self.plot_type is not None:
            raise ValueError("Only chart analysis may specify a plot type")
        if self.kind != "chart" and self.series:
            raise ValueError("Only chart analysis may contain numeric series")
        if self.kind != "flowchart" and (self.nodes or self.edges):
            raise ValueError("Only flowcharts may contain nodes and edges")
        identities = [node.id for node in self.nodes]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate flowchart node IDs")
        if any(
            edge.source not in identities or edge.target not in identities for edge in self.edges
        ):
            raise ValueError("Flowchart edge refers to a missing node")
        return self


class VisualDescription(SchemaModel):
    text: str
    provenance: Provenance
    analysis: VisualAnalysis | None = None


class VisualQualityReview(SchemaModel):
    status: Literal["unverified", "model_reviewed", "source_correlated", "needs_review"] = (
        "unverified"
    )
    reviewer: str | None = None
    adjustments: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ImageElement(ElementBase):
    type: Literal["image"] = "image"
    asset: AssetReference
    caption: str | None = None
    description: VisualDescription | None = None
    quality_review: VisualQualityReview | None = None


class ChartElement(ElementBase):
    type: Literal["chart"] = "chart"
    asset: AssetReference
    caption: str | None = None
    description: VisualDescription | None = None
    series: list[ChartSeries] = Field(default_factory=list)
    series_provenance: Provenance | None = None
    quality_review: VisualQualityReview | None = None


DocumentElement = Annotated[
    TextElement | HeadingElement | FormulaElement | TableElement | ImageElement | ChartElement,
    Field(discriminator="type"),
]


class PreparationSettings(SchemaModel):
    dpi: int = Field(ge=36, le=600)
    renderer: str


class Document(SchemaModel):
    schema_version: Literal["1.0"] = "1.0"
    document_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_filename: str
    title: str | None = None
    status: Literal["prepared", "parsing", "parsed"] = "prepared"
    preparation: PreparationSettings
    pages: list[Page] = Field(min_length=1)
    # Array order is the canonical reading order, including content across pages.
    elements: list[DocumentElement] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_references(self) -> Self:
        if [page.page_number for page in self.pages] != list(range(1, len(self.pages) + 1)):
            raise ValueError("Pages must be contiguous and ordered, starting at 1")
        seen: dict[str, DocumentElement] = {}
        for element in self.elements:
            if element.id in seen:
                raise ValueError("Duplicate element ID")
            if element.parent_id is not None:
                parent = seen.get(element.parent_id)
                if not isinstance(parent, HeadingElement):
                    raise ValueError("Parent must reference an earlier heading")
                if isinstance(element, HeadingElement) and element.level <= parent.level:
                    raise ValueError("Child heading level must exceed parent heading level")
            if any(region.page_number > len(self.pages) for region in element.source_regions):
                raise ValueError("Source region references a missing page")
            seen[element.id] = element
        for element in self.elements:
            if isinstance(element, TableElement):
                for identity in element.footnote_element_ids:
                    if not isinstance(seen.get(identity), TextElement):
                        raise ValueError("Table footnote must reference a text element")
                if element.caption_element_id is not None:
                    caption = seen.get(element.caption_element_id)
                    if not isinstance(caption, TextElement) or caption.type != "caption":
                        raise ValueError("Table caption must reference a caption element")
                if element.continuation_index and not element.logical_table_id:
                    raise ValueError("Table continuation requires a logical table ID")
        if self.status == "prepared" and self.elements:
            raise ValueError("Prepared documents cannot contain parsed elements")
        return self
