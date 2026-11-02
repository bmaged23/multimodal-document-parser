"""Versioned request contracts; artifact IDs never accept server filesystem paths."""

from typing import Annotated, Literal

from pydantic import Field

from multimodal_doc_parser.document_schema.document import SchemaModel
from multimodal_doc_parser.rag_preparation.chunk_schema import ChunkSettings

ArtifactId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]


class PreparationJob(SchemaModel):
    source_artifact_id: ArtifactId
    dpi: int = Field(default=144, ge=36, le=300)
    max_pages: int = Field(default=200, ge=1, le=1000)


class ParsingJob(SchemaModel):
    source_artifact_id: ArtifactId
    backend: Literal["paddleocr", "dots-mocr"] = "paddleocr"
    enrich_visuals: bool = True


class ChunkingJob(SchemaModel):
    source_artifact_id: ArtifactId
    settings: ChunkSettings = Field(default_factory=ChunkSettings)


class IndexingJob(SchemaModel):
    source_artifact_ids: list[ArtifactId] = Field(min_length=1, max_length=100)
    batch_size: int = Field(default=2, ge=1, le=8)


class SearchRequest(SchemaModel):
    index_artifact_id: ArtifactId
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=30)
    context_tokens: int = Field(default=2400, ge=128, le=12000)
    mode: Literal["hybrid", "keyword", "dense"] = "hybrid"
    expand_parents: bool = True


class RAGRequest(SchemaModel):
    index_artifact_id: ArtifactId
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=15)
    context_tokens: int = Field(default=2400, ge=512, le=6000)
    allow_query_rewrite: bool = True


class PipelineJob(PreparationJob):
    backend: Literal["paddleocr", "dots-mocr"] = "paddleocr"
    chunk_settings: ChunkSettings = Field(default_factory=ChunkSettings)


class RAGDecision(SchemaModel):
    action: Literal["answer", "rewrite", "abstain"]
    query: str = Field(default="", max_length=4000)
    reason: str = Field(max_length=1000)


class RAGAnswer(SchemaModel):
    answer: str = Field(min_length=1, max_length=12000)
    citations: list[str] = Field(default_factory=list, max_length=30)
    insufficient_evidence: bool


JOB_REQUESTS = {
    "documents": PreparationJob,
    "parsing": ParsingJob,
    "chunking": ChunkingJob,
    "indexing": IndexingJob,
    "rag": RAGRequest,
    "pipeline": PipelineJob,
}
