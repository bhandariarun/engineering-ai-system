from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    session_id: str = "default"


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float


class AssistantResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    tool_calls: list[dict[str, Any]] = []
    provider: str
    cached: bool = False
    degraded: bool = False


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    provider: str
    indexed_chunks: int
