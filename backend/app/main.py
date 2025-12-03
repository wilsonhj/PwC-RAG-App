from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import get_config, RetrievalStrategy
from .rag import answer_question, get_recent_metrics, get_collection_stats


config = get_config()

app = FastAPI(
    title="PWC RAG API",
    version="0.2.0",
    description="Modular RAG pipeline with multi-model routing and observability",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str
    use_routing: bool = True
    top_k: Optional[int] = None
    strategy: Optional[str] = None  # "semantic", "hybrid", "adaptive"


class MetricsResponse(BaseModel):
    """Metrics for a single query."""
    model_tier: str
    model_used: str
    retrieval_strategy: str
    num_chunks_retrieved: int
    latency_retrieve_ms: float
    latency_generate_ms: float
    latency_total_ms: float


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    citations: list[str]
    metrics: Optional[MetricsResponse] = None


class StatsResponse(BaseModel):
    """Response model for collection statistics."""
    total_documents: int
    collection_name: str
    distance_metric: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def read_root():
    """Health check endpoint."""
    return {"message": "PWC RAG API is running", "version": "0.2.0"}


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system.
    
    - **question**: The question to answer
    - **use_routing**: Whether to use the router to select model tier (default: True)
    - **top_k**: Override for number of chunks to retrieve
    - **strategy**: Override retrieval strategy ("semantic", "hybrid", "adaptive")
    """
    # Parse strategy if provided
    strategy = None
    if request.strategy:
        try:
            strategy = RetrievalStrategy(request.strategy)
        except ValueError:
            pass  # Use default
    
    answer, citations, metrics = answer_question(
        request.question,
        use_routing=request.use_routing,
        top_k=request.top_k,
        strategy=strategy,
    )
    
    metrics_response = None
    if metrics:
        metrics_response = MetricsResponse(
            model_tier=metrics.model_tier,
            model_used=metrics.model_used,
            retrieval_strategy=metrics.retrieval_strategy,
            num_chunks_retrieved=metrics.num_chunks_retrieved,
            latency_retrieve_ms=round(metrics.latency_retrieve_ms, 2),
            latency_generate_ms=round(metrics.latency_generate_ms, 2),
            latency_total_ms=round(metrics.latency_total_ms, 2),
        )
    
    return QueryResponse(
        answer=answer,
        citations=citations,
        metrics=metrics_response,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get vector collection statistics."""
    stats = get_collection_stats()
    return StatsResponse(**stats)


@app.get("/metrics")
async def get_metrics():
    """Get recent query metrics for observability."""
    metrics = get_recent_metrics()
    return {
        "count": len(metrics),
        "metrics": [
            {
                "query": m.query[:50] + "..." if len(m.query) > 50 else m.query,
                "model_tier": m.model_tier,
                "model_used": m.model_used,
                "retrieval_strategy": m.retrieval_strategy,
                "num_chunks": m.num_chunks_retrieved,
                "latency_ms": round(m.latency_total_ms, 2),
            }
            for m in metrics[-10:]  # Last 10
        ],
    }


@app.get("/config")
async def get_current_config():
    """Get current configuration (for debugging)."""
    return {
        "embedding_model": config.embedding.model,
        "llm_models": {k.value: v for k, v in config.llm.models.items()},
        "retrieval": {
            "strategy": config.retrieval.strategy.value,
            "top_k": config.retrieval.top_k,
            "similarity_threshold": config.retrieval.similarity_threshold,
        },
        "chunking": {
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
        },
    }
