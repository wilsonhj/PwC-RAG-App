"""
Retriever module with semantic, BM25, and hybrid search strategies.

Strategies:
- SEMANTIC: Pure vector similarity via ChromaDB
- HYBRID: Weighted combination of semantic + BM25 keyword search
- ADAPTIVE: Router model selects strategy based on query characteristics
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import time

from rank_bm25 import BM25Okapi

from ..config import get_config, RetrievalStrategy


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    strategy: str = "semantic"  # Which strategy produced this result


@dataclass
class RetrievalResponse:
    """Response from a retrieval operation."""
    results: List[RetrievalResult]
    strategy_used: RetrievalStrategy
    latency_ms: float
    query: str
    
    @property
    def ids(self) -> List[str]:
        return [r.id for r in self.results]
    
    @property
    def texts(self) -> List[str]:
        return [r.text for r in self.results]


class Retriever(ABC):
    """Abstract base class for retrieval strategies."""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> RetrievalResponse:
        """Retrieve relevant documents for a query."""
        pass


class SemanticRetriever(Retriever):
    """
    Pure vector similarity retrieval using embeddings.
    
    Uses ChromaDB for vector storage and Ollama for embeddings.
    """
    
    def __init__(
        self,
        collection,
        embed_fn: Callable[[str], Tuple[List[float], float]],
        similarity_threshold: float = 0.5,
    ):
        """
        Initialize semantic retriever.
        
        Args:
            collection: ChromaDB collection
            embed_fn: Function that returns (embedding, latency_ms)
            similarity_threshold: Minimum similarity score to include
        """
        self.collection = collection
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
    
    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> RetrievalResponse:
        """Retrieve using vector similarity."""
        start = time.time()
        
        # Get query embedding
        query_embedding, _ = self.embed_fn(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        
        # Build results
        retrieval_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            score = 1 - results["distances"][0][i]  # Convert distance to similarity
            
            if score >= self.similarity_threshold:
                retrieval_results.append(RetrievalResult(
                    id=doc_id,
                    text=results["documents"][0][i],
                    score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    strategy="semantic",
                ))
        
        latency_ms = (time.time() - start) * 1000
        
        return RetrievalResponse(
            results=retrieval_results,
            strategy_used=RetrievalStrategy.SEMANTIC,
            latency_ms=latency_ms,
            query=query,
        )


class BM25Retriever(Retriever):
    """
    BM25 keyword-based retrieval.
    
    Maintains an in-memory BM25 index over document texts.
    """
    
    def __init__(
        self,
        documents: List[dict],
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'
            tokenizer: Function to tokenize text (default: simple whitespace split)
        """
        self.documents = documents
        self.tokenizer = tokenizer or self._default_tokenizer
        
        # Build BM25 index
        self._build_index()
    
    def _default_tokenizer(self, text: str) -> List[str]:
        """Simple lowercase whitespace tokenizer."""
        return text.lower().split()
    
    def _build_index(self):
        """Build BM25 index from documents."""
        tokenized_docs = [
            self.tokenizer(doc["text"]) for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def update_documents(self, documents: List[dict]):
        """Update the document corpus and rebuild index."""
        self.documents = documents
        self._build_index()
    
    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> RetrievalResponse:
        """Retrieve using BM25 scoring."""
        start = time.time()
        
        # Tokenize query
        query_tokens = self.tokenizer(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]
        
        # Build results
        retrieval_results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                doc = self.documents[idx]
                retrieval_results.append(RetrievalResult(
                    id=doc["id"],
                    text=doc["text"],
                    score=float(scores[idx]),
                    metadata=doc.get("metadata", {}),
                    strategy="bm25",
                ))
        
        latency_ms = (time.time() - start) * 1000
        
        return RetrievalResponse(
            results=retrieval_results,
            strategy_used=RetrievalStrategy.HYBRID,  # BM25 is part of hybrid
            latency_ms=latency_ms,
            query=query,
        )


class HybridRetriever(Retriever):
    """
    Hybrid retrieval combining semantic and BM25 search.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results from both strategies.
    """
    
    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = 60,  # RRF constant
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            semantic_retriever: Vector similarity retriever
            bm25_retriever: BM25 keyword retriever
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for BM25 scores (0-1)
            rrf_k: Reciprocal Rank Fusion constant
        """
        self.semantic = semantic_retriever
        self.bm25 = bm25_retriever
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
    
    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> RetrievalResponse:
        """Retrieve using hybrid RRF fusion."""
        start = time.time()
        
        # Get results from both retrievers (fetch more for fusion)
        fetch_k = k * 2
        semantic_response = self.semantic.retrieve(query, k=fetch_k)
        bm25_response = self.bm25.retrieve(query, k=fetch_k)
        
        # Build RRF scores
        rrf_scores = {}
        
        # Add semantic scores
        for rank, result in enumerate(semantic_response.results):
            rrf_score = self.semantic_weight / (self.rrf_k + rank + 1)
            if result.id in rrf_scores:
                rrf_scores[result.id]["score"] += rrf_score
                rrf_scores[result.id]["strategies"].append("semantic")
            else:
                rrf_scores[result.id] = {
                    "result": result,
                    "score": rrf_score,
                    "strategies": ["semantic"],
                }
        
        # Add BM25 scores
        for rank, result in enumerate(bm25_response.results):
            rrf_score = self.keyword_weight / (self.rrf_k + rank + 1)
            if result.id in rrf_scores:
                rrf_scores[result.id]["score"] += rrf_score
                rrf_scores[result.id]["strategies"].append("bm25")
            else:
                rrf_scores[result.id] = {
                    "result": result,
                    "score": rrf_score,
                    "strategies": ["bm25"],
                }
        
        # Sort by RRF score and take top-k
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:k]
        
        # Build final results
        retrieval_results = []
        for item in sorted_results:
            result = item["result"]
            retrieval_results.append(RetrievalResult(
                id=result.id,
                text=result.text,
                score=item["score"],
                metadata={
                    **result.metadata,
                    "strategies": item["strategies"],
                },
                strategy="hybrid",
            ))
        
        latency_ms = (time.time() - start) * 1000
        
        return RetrievalResponse(
            results=retrieval_results,
            strategy_used=RetrievalStrategy.HYBRID,
            latency_ms=latency_ms,
            query=query,
        )


class AdaptiveRetriever(Retriever):
    """
    Adaptive retrieval that selects strategy based on query characteristics.
    
    Uses heuristics or a router model to decide between semantic and hybrid.
    """
    
    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        hybrid_retriever: HybridRetriever,
        route_fn: Optional[Callable[[str], RetrievalStrategy]] = None,
    ):
        """
        Initialize adaptive retriever.
        
        Args:
            semantic_retriever: For simple factual queries
            hybrid_retriever: For keyword-heavy or complex queries
            route_fn: Optional function to classify query -> strategy
        """
        self.semantic = semantic_retriever
        self.hybrid = hybrid_retriever
        self.route_fn = route_fn or self._default_router
    
    def _default_router(self, query: str) -> RetrievalStrategy:
        """
        Simple heuristic router based on query characteristics.
        
        Uses hybrid for:
        - Short queries (likely keyword searches)
        - Queries with specific technical terms
        - Queries with quotes (exact match intent)
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        # Short queries → hybrid (keyword-heavy)
        if len(words) <= 3:
            return RetrievalStrategy.HYBRID
        
        # Quoted phrases → hybrid (exact match intent)
        if '"' in query or "'" in query:
            return RetrievalStrategy.HYBRID
        
        # Question words → semantic (natural language)
        question_words = {"what", "how", "why", "when", "where", "who", "which"}
        if words[0] in question_words:
            return RetrievalStrategy.SEMANTIC
        
        # Default to semantic for longer, natural queries
        return RetrievalStrategy.SEMANTIC
    
    def retrieve(
        self,
        query: str,
        k: int = 3,
    ) -> RetrievalResponse:
        """Retrieve using adaptively selected strategy."""
        start = time.time()
        
        # Route to appropriate strategy
        strategy = self.route_fn(query)
        
        if strategy == RetrievalStrategy.HYBRID:
            response = self.hybrid.retrieve(query, k=k)
        else:
            response = self.semantic.retrieve(query, k=k)
        
        # Update latency to include routing time
        total_latency = (time.time() - start) * 1000
        
        return RetrievalResponse(
            results=response.results,
            strategy_used=RetrievalStrategy.ADAPTIVE,
            latency_ms=total_latency,
            query=query,
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def create_retriever(
    strategy: RetrievalStrategy,
    collection,
    embed_fn: Callable[[str], Tuple[List[float], float]],
    documents: List[dict],
    config=None,
) -> Retriever:
    """
    Factory function to create the appropriate retriever.
    
    Args:
        strategy: Which retrieval strategy to use
        collection: ChromaDB collection for semantic search
        embed_fn: Embedding function
        documents: Document corpus for BM25
        config: Optional config override
        
    Returns:
        Configured Retriever instance
    """
    cfg = config or get_config()
    
    # Always create semantic retriever
    semantic = SemanticRetriever(
        collection=collection,
        embed_fn=embed_fn,
        similarity_threshold=cfg.retrieval.similarity_threshold,
    )
    
    if strategy == RetrievalStrategy.SEMANTIC:
        return semantic
    
    # Create BM25 for hybrid/adaptive
    bm25 = BM25Retriever(documents=documents)
    
    if strategy == RetrievalStrategy.HYBRID:
        return HybridRetriever(
            semantic_retriever=semantic,
            bm25_retriever=bm25,
            semantic_weight=cfg.retrieval.semantic_weight,
            keyword_weight=cfg.retrieval.keyword_weight,
        )
    
    # Adaptive
    hybrid = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=bm25,
        semantic_weight=cfg.retrieval.semantic_weight,
        keyword_weight=cfg.retrieval.keyword_weight,
    )
    
    return AdaptiveRetriever(
        semantic_retriever=semantic,
        hybrid_retriever=hybrid,
    )
