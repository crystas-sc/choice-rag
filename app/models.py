"""models.py — request/response schemas."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=50)
    min_score: float = Field(0.0, ge=0.0)
    ollama_url: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_gen_model: Optional[str] = None
    skip_decision: bool = False


class SearchResult(BaseModel):
    rank: int
    score: float
    id: str
    question: str
    answer: str
    source: str
    keywords: str
    variants: List[str] = Field(default_factory=list)
    original_text: Optional[str] = None
    quote: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    document: Optional[str] = None
    answer_source: Optional[str] = "corpus"


class DecisionOut(BaseModel):
    decision: str  # "Just top 1" | "Just top 2" | "both top1 and top2" | "none" | "generated"
    label: str     # "Just top 1", "Just top 2", "both top1 and top2", "Generated on the fly", or "No appropriate answer found, Ask an agent"
    model: str
    url: str
    raw_output: Optional[str] = None
    error: Optional[str] = None
    generated_answer: Optional[str] = None
    answer_source: Optional[str] = None
    generation_ms: Optional[float] = None
    chunk_id: Optional[str] = None


class DecisionRequest(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    ollama_url: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_gen_model: Optional[str] = None
    synthesize_if_none: bool = False


class DecisionResponse(BaseModel):
    decision: DecisionOut
    decision_ms: float
    decision_sec: Optional[float] = None


class GenerateRequest(BaseModel):
    query: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    ollama_url: Optional[str] = None
    ollama_gen_model: Optional[str] = None


class GenerateResponse(BaseModel):
    query: str
    generated_answer: Optional[str] = None
    chunk_id: Optional[str] = None
    answer_source: str = "generated_on_the_fly"
    generator_model: str = ""
    generation_ms: float = 0.0
    generation_sec: float = 0.0
    error: Optional[str] = None
    fallback_to_agent: bool = False


class SearchResponse(BaseModel):
    query: str
    returned: int
    elapsed_ms: float
    elapsed_sec: Optional[float] = None
    retrieval_ms: float = 0.0
    retrieval_sec: Optional[float] = None
    decision_ms: Optional[float] = None
    decision_sec: Optional[float] = None
    generation_ms: Optional[float] = None
    generation_sec: Optional[float] = None
    decision: Optional[DecisionOut] = None
    results: List[SearchResult]


class HealthOut(BaseModel):
    status: str
    active_chunks: int
    fts_rows: int
    documents: List[Dict[str, Any]]


class IngestResult(BaseModel):
    document_id: int
    version: int
    chunks: int
    skipped: bool = False


class ActivateRequest(BaseModel):
    filename: str
    version: int


class ChunkOut(BaseModel):
    id: str
    question: str
    variants: List[str] = Field(default_factory=list)
    answer: str
    source: str
    original_text: str
    keywords: str
    evidence: Optional[Dict[str, Any]] = None
    document: Optional[str] = None
