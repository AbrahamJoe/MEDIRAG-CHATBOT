"""Pydantic schemas for data validation."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """Represents extracted content from a single PDF page."""

    page_number: int
    text: str


class DocumentChunk(BaseModel):
    """Represents a text chunk with metadata."""

    chunk_id: str
    text: str
    source_file: str
    page_number: int


class VectorRecord(BaseModel):
    """Represents a vector to be stored in Pinecone."""

    id: str
    values: list[float]
    metadata: dict


class RetrievedChunk(BaseModel):
    """Represents a chunk retrieved from Pinecone."""

    text: str
    source_file: str
    page_number: int
    score: float


class ChatMessage(BaseModel):
    """Represents a single chat message."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    sources: Optional[list[RetrievedChunk]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class QueryResponse(BaseModel):
    """Represents the full response to a user query."""

    answer: str
    sources: list[RetrievedChunk]
    query: str


class DocumentStats(BaseModel):
    """Statistics about ingested documents."""

    total_pdfs: int = 0
    total_chunks: int = 0
    total_vectors: int = 0
    file_names: list[str] = Field(default_factory=list)
