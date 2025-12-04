"""
Pipeline modules for the RAG system.

Modular components:
- chunker: Document chunking strategies
- retriever: Retrieval strategies (semantic, hybrid, adaptive)
- agent: LangGraph-based agentic workflow
- tokenizer: Token counting with tiktoken
- logger: Structured JSON logging
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
from .tokenizer import (
    TokenCounter,
    TokenUsage,
    get_token_counter,
    count_tokens,
    count_usage,
)
from .logger import (
    StructuredLogger,
    LogLevel,
    LogEntry,
    get_logger,
    configure_logger,
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
    # Tokenizer
    "TokenCounter",
    "TokenUsage",
    "get_token_counter",
    "count_tokens",
    "count_usage",
    # Logger
    "StructuredLogger",
    "LogLevel",
    "LogEntry",
    "get_logger",
    "configure_logger",
]
