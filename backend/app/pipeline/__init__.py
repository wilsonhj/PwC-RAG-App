"""
Pipeline modules for the RAG system.

Modular components:
- chunker: Document chunking strategies
- embedder: Embedding abstraction
- retriever: Retrieval strategies (semantic, hybrid, adaptive)
- generator: LLM generation with token tracking
"""

from .chunker import Chunker, RecursiveChunker, Chunk

__all__ = ["Chunker", "RecursiveChunker", "Chunk"]
