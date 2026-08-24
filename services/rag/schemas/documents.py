from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source: str | None = None
    content_type: str = "text/plain"
    metadata: dict[str, Any] | None = None


class ChunkInfo(BaseModel):
    chunk_index: int
    content_snippet: str = Field(default="", description="First 200 chars of chunk content")


class DocumentResponse(BaseModel):
    id: str
    title: str
    source: str | None = None
    content_type: str
    status: str
    chunks_count: int
    metadata: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: dict[str, Any] | None = None
    language: str | None = Field(
        default=None,
        description=(
            "Query language hint (e.g. 'hi-IN'). When set and non-English, "
            "the query is translated to the index language before retrieval."
        ),
    )


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    chunk_index: int
    content: str
    score: float
    metadata: dict[str, Any] | None = None


class Citation(BaseModel):
    document_id: str
    document_title: str
    chunk_index: int
    content: str = Field(default="", description="Snippet of the cited chunk")
    score: float


class Source(BaseModel):
    document_id: str
    document_title: str
    snippet: str = Field(default="", description="Short snippet from the cited chunk")
    score: float = 0.0


class AnswerResult(BaseModel):
    answer: str = Field(description="Natural-language answer generated from retrieved chunks")
    sources: list[Source] = Field(
        default_factory=list,
        description="Documents the answer drew from",
    )


class QueryResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_found: int
    citations: list[Citation] = Field(
        default_factory=list,
        description="Attributable sources for the answer",
    )
    answer: str | None = Field(
        default=None,
        description="Natural-language answer synthesized from retrieved chunks",
    )
    sources: list[Source] = Field(
        default_factory=list,
        description="Documents the answer drew from",
    )
    refined_query: str | None = Field(
        default=None,
        description="LLM-refined search query when agentic refinement ran",
    )
    translated_query: str | None = Field(
        default=None,
        description="Query translated to the index language before retrieval",
    )
    language: str | None = None
