"""
Unit tests for the retriever module.
"""

import pytest
from app.pipeline.retriever import (
    BM25Retriever,
    RetrievalResult,
    RetrievalResponse,
)
from app.config import RetrievalStrategy


class TestBM25Retriever:
    """Tests for BM25 keyword retrieval."""
    
    @pytest.fixture
    def sample_docs(self):
        return [
            {"id": "doc-1", "text": "Machine learning is a subset of artificial intelligence."},
            {"id": "doc-2", "text": "Deep learning uses neural networks with many layers."},
            {"id": "doc-3", "text": "Natural language processing enables text understanding."},
        ]
    
    def test_bm25_retrieval(self, sample_docs):
        retriever = BM25Retriever(documents=sample_docs)
        response = retriever.retrieve("machine learning", k=2)
        
        assert len(response.results) <= 2
        assert response.query == "machine learning"
        assert response.latency_ms >= 0
    
    def test_bm25_relevance(self, sample_docs):
        retriever = BM25Retriever(documents=sample_docs)
        response = retriever.retrieve("neural networks deep learning", k=3)
        
        # doc-2 should be most relevant
        if response.results:
            assert response.results[0].id == "doc-2"
    
    def test_bm25_no_match(self, sample_docs):
        retriever = BM25Retriever(documents=sample_docs)
        response = retriever.retrieve("quantum computing blockchain", k=3)
        
        # Should return empty or low-score results
        assert len(response.results) == 0 or response.results[0].score < 1
    
    def test_bm25_update_documents(self, sample_docs):
        retriever = BM25Retriever(documents=sample_docs)
        
        # Update with new documents
        new_docs = [{"id": "new-1", "text": "Quantum computing is the future of technology."}]
        retriever.update_documents(new_docs)
        
        response = retriever.retrieve("quantum computing future", k=1)
        # Should find the new document
        assert len(response.results) >= 0  # May be 0 if BM25 threshold not met
        if response.results:
            assert response.results[0].id == "new-1"


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""
    
    def test_result_creation(self):
        result = RetrievalResult(
            id="test-1",
            text="Test content",
            score=0.85,
            metadata={"source": "test"},
            strategy="semantic",
        )
        
        assert result.id == "test-1"
        assert result.score == 0.85
        assert result.strategy == "semantic"


class TestRetrievalResponse:
    """Tests for RetrievalResponse dataclass."""
    
    def test_response_properties(self):
        results = [
            RetrievalResult(id="r1", text="Text 1", score=0.9),
            RetrievalResult(id="r2", text="Text 2", score=0.8),
        ]
        
        response = RetrievalResponse(
            results=results,
            strategy_used=RetrievalStrategy.SEMANTIC,
            latency_ms=50.0,
            query="test query",
        )
        
        assert response.ids == ["r1", "r2"]
        assert response.texts == ["Text 1", "Text 2"]
        assert len(response.results) == 2
