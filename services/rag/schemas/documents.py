from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source: Optional[str] = None
    content_type: str = "text/plain"
    metadata: Optional[dict[str, Any]] = None


class ChunkInfo(BaseModel):
    chunk_index: int
    content_snippet: str = Field(default="", description="First 200 chars of chunk content")


class DocumentResponse(BaseModel):
    id: str
    title: str
    source: Optional[str] = None
    content_type: str
    status: str
    chunks_count: int
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Optional[dict[str, Any]] = None


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    chunk_index: int
    content: str
    score: float
    metadata: Optional[dict[str, Any]] = None


class QueryResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_found: int
