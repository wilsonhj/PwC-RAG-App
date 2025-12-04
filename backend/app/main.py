from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import get_config, RetrievalStrategy
from .rag import (
    answer_question,
    answer_with_agent,
    get_recent_metrics,
    get_collection_stats,
    ingest_text,
    ingest_documents,
    delete_document,
    clear_all_documents,
    get_all_documents,
    IngestResult,
)
from .pipeline.agent import AgentResult


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


# Ingestion models
class IngestTextRequest(BaseModel):
    """Request model for ingesting raw text."""
    text: str
    source_id: Optional[str] = None
    metadata: Optional[dict] = None


class IngestDocumentsRequest(BaseModel):
    """Request model for ingesting multiple documents."""
    documents: List[dict]  # Each with 'text', optional 'source_id', 'metadata'


class IngestResponse(BaseModel):
    """Response model for ingestion."""
    source_id: str
    chunks_created: int
    chunks_indexed: int
    total_documents: int


class BulkIngestResponse(BaseModel):
    """Response model for bulk ingestion."""
    documents_processed: int
    total_chunks_created: int
    total_chunks_indexed: int
    total_documents: int


# Agent models
class AgentQueryRequest(BaseModel):
    """Request model for agent-based queries."""
    question: str


class AgentQueryResponse(BaseModel):
    """Response model for agent-based queries."""
    answer: str
    citations: List[str]
    complexity: str
    retrieval_strategy: str
    steps_executed: List[str]
    latency_ms: float
    model_used: str
    refinement_count: int


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


@app.post("/agent", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
    """
    Query using the LangGraph agent.
    
    The agent uses a state machine workflow:
    1. **classify**: Determine query complexity (simple/complex)
    2. **retrieve**: Fetch relevant passages using optimal strategy
    3. **generate**: Produce answer with appropriate model tier
    4. **refine** (optional): Improve answer if uncertain
    
    Returns detailed execution metadata including steps executed.
    """
    try:
        result = answer_with_agent(request.question)
        
        return AgentQueryResponse(
            answer=result.answer,
            citations=result.citations,
            complexity=result.complexity,
            retrieval_strategy=result.retrieval_strategy,
            steps_executed=result.steps_executed,
            latency_ms=round(result.latency_ms, 2),
            model_used=result.model_used,
            refinement_count=result.refinement_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


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


# ---------------------------------------------------------------------------
# Document Ingestion Endpoints
# ---------------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse)
async def ingest_text_endpoint(request: IngestTextRequest):
    """
    Ingest raw text into the RAG system.
    
    The text will be:
    1. Chunked using the recursive chunker
    2. Embedded using the configured embedding model
    3. Stored in ChromaDB for retrieval
    
    - **text**: The raw text to ingest
    - **source_id**: Optional identifier for the source document
    - **metadata**: Optional metadata to attach to all chunks
    """
    result = ingest_text(
        text=request.text,
        source_id=request.source_id,
        metadata=request.metadata,
    )
    
    return IngestResponse(
        source_id=result.source_id,
        chunks_created=result.chunks_created,
        chunks_indexed=result.chunks_indexed,
        total_documents=result.total_documents,
    )


@app.post("/ingest/bulk", response_model=BulkIngestResponse)
async def ingest_bulk_endpoint(request: IngestDocumentsRequest):
    """
    Ingest multiple documents into the RAG system.
    
    Each document should have:
    - **text**: The raw text (required)
    - **source_id**: Optional identifier
    - **metadata**: Optional metadata dict
    """
    results = ingest_documents(request.documents)
    
    return BulkIngestResponse(
        documents_processed=len(results),
        total_chunks_created=sum(r.chunks_created for r in results),
        total_chunks_indexed=sum(r.chunks_indexed for r in results),
        total_documents=results[-1].total_documents if results else 0,
    )


@app.get("/documents")
async def list_documents():
    """List all documents in the system."""
    docs = get_all_documents()
    return {
        "count": len(docs),
        "documents": [
            {
                "id": d["id"],
                "text_preview": d["text"][:100] + "..." if len(d["text"]) > 100 else d["text"],
                "metadata": d.get("metadata", {}),
            }
            for d in docs
        ],
    }


@app.delete("/documents/{doc_id}")
async def delete_document_endpoint(doc_id: str):
    """Delete a specific document by ID."""
    success = delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"message": f"Document {doc_id} deleted", "success": True}


@app.delete("/documents")
async def clear_documents_endpoint():
    """Delete all documents from the system."""
    count = clear_all_documents()
    return {"message": f"Deleted {count} documents", "count": count}
