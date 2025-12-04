"""
Pipeline modules for the RAG system.

Modular components:
- chunker: Document chunking strategies
- retriever: Retrieval strategies (semantic, hybrid, adaptive)
- agent: LangGraph-based agentic workflow
- embedder: Embedding abstraction (planned)
- generator: LLM generation with token tracking (planned)
"""

from .chunker import Chunker, RecursiveChunker, Chunk
from .retriever import (
    Retriever,
    SemanticRetriever,
    BM25Retriever,
    HybridRetriever,
    AdaptiveRetriever,
    RetrievalResult,
    RetrievalResponse,
    create_retriever,
)
from .agent import (
    RAGAgent,
    AgentConfig,
    AgentResult,
    AgentState,
    build_rag_agent,
)

__all__ = [
    # Chunker
    "Chunker",
    "RecursiveChunker",
    "Chunk",
    # Retriever
    "Retriever",
    "SemanticRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "AdaptiveRetriever",
    "RetrievalResult",
    "RetrievalResponse",
    "create_retriever",
    # Agent
    "RAGAgent",
    "AgentConfig",
    "AgentResult",
    "AgentState",
    "build_rag_agent",
]
