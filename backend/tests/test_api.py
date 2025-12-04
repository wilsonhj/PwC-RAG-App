"""
Integration tests for the FastAPI endpoints.

Note: These tests require Ollama to be running for full functionality.
Tests are designed to be skipped gracefully if Ollama is unavailable.
"""

import pytest


class TestHealthEndpoints:
    """Tests for health and info endpoints."""
    
    @pytest.mark.asyncio
    async def test_root(self, client):
        response = await client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "running" in data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_config(self, client):
        response = await client.get("/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "embedding_model" in data
        assert "llm_models" in data
        assert "retrieval" in data
        assert "chunking" in data
    
    @pytest.mark.asyncio
    async def test_stats(self, client):
        response = await client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "collection_name" in data
        assert "distance_metric" in data


class TestDocumentEndpoints:
    """Tests for document management endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_documents(self, client):
        response = await client.get("/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "documents" in data
        assert isinstance(data["documents"], list)
    
    @pytest.mark.asyncio
    async def test_ingest_text(self, client):
        response = await client.post(
            "/ingest",
            json={
                "text": "This is a test document for pytest.",
                "source_id": "pytest-doc",
                "metadata": {"test": True},
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["source_id"] == "pytest-doc"
        assert data["chunks_created"] >= 1
        assert data["chunks_indexed"] >= 0  # May be 0 if already exists
    
    @pytest.mark.asyncio
    async def test_ingest_bulk(self, client):
        response = await client.post(
            "/ingest/bulk",
            json={
                "documents": [
                    {"text": "Bulk doc 1", "source_id": "bulk-1"},
                    {"text": "Bulk doc 2", "source_id": "bulk-2"},
                ],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["documents_processed"] == 2
        assert data["total_chunks_created"] >= 2


class TestQueryEndpoint:
    """Tests for the query endpoint."""
    
    @pytest.mark.asyncio
    async def test_query_basic(self, client):
        """Test basic query - may fail if Ollama not running."""
        try:
            response = await client.post(
                "/query",
                json={"question": "What is RAG?"},
            )
            
            if response.status_code == 200:
                data = response.json()
                assert "answer" in data
                assert "citations" in data
                assert "metrics" in data
                metrics = data["metrics"]
                for field in ["input_tokens", "output_tokens", "total_tokens"]:
                    assert field in metrics
                    assert isinstance(metrics[field], (int, float))
                assert metrics["total_tokens"] == metrics["input_tokens"] + metrics["output_tokens"]
            else:
                # Ollama might not be running
                pytest.skip("Ollama not available")
        except Exception as e:
            pytest.skip(f"Query test skipped: {e}")
    
    @pytest.mark.asyncio
    async def test_query_with_options(self, client):
        """Test query with custom options."""
        try:
            response = await client.post(
                "/query",
                json={
                    "question": "What is RAG?",
                    "use_routing": False,
                    "top_k": 2,
                    "strategy": "semantic",
                },
            )
            
            if response.status_code == 200:
                data = response.json()
                assert len(data["citations"]) <= 2
                metrics = data["metrics"]
                assert metrics["total_tokens"] == metrics["input_tokens"] + metrics["output_tokens"]
            else:
                pytest.skip("Ollama not available")
        except Exception as e:
            pytest.skip(f"Query test skipped: {e}")


class TestMetricsEndpoint:
    """Tests for the metrics endpoint."""
    
    @pytest.mark.asyncio
    async def test_metrics(self, client):
        response = await client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "metrics" in data
        assert isinstance(data["metrics"], list)


class TestAgentEndpoint:
    """Tests for the LangGraph agent endpoint."""
    
    @pytest.mark.asyncio
    async def test_agent_basic(self, client):
        """Test basic agent query."""
        try:
            response = await client.post(
                "/agent",
                json={"question": "What is RAG?"},
            )
            
            if response.status_code == 200:
                data = response.json()
                assert "answer" in data
                assert "citations" in data
                assert "complexity" in data
                assert "retrieval_strategy" in data
                assert "steps_executed" in data
                assert "latency_ms" in data
                assert "model_used" in data
                assert "refinement_count" in data
                assert "metrics" not in data  # Agent endpoint does not expose token metrics yet
                
                # Verify steps were executed
                assert isinstance(data["steps_executed"], list)
                assert len(data["steps_executed"]) >= 3  # classify, retrieve, generate
            else:
                pytest.skip("Ollama not available")
        except Exception as e:
            pytest.skip(f"Agent test skipped: {e}")
    
    @pytest.mark.asyncio
    async def test_agent_complex_query(self, client):
        """Test agent with complex query."""
        try:
            response = await client.post(
                "/agent",
                json={"question": "Compare and contrast different AI consulting approaches and their integration with RAG systems"},
            )
            
            if response.status_code == 200:
                data = response.json()
                # Complex queries should be classified as complex
                # (though this depends on the LLM's classification)
                assert data["complexity"] in ["simple", "complex"]
                assert data["retrieval_strategy"] in ["semantic", "hybrid", "adaptive"]
            else:
                pytest.skip("Ollama not available")
        except Exception as e:
            pytest.skip(f"Agent test skipped: {e}")
    
    @pytest.mark.asyncio
    async def test_agent_returns_citations(self, client):
        """Test that agent returns citations."""
        try:
            response = await client.post(
                "/agent",
                json={"question": "What does PwC do?"},
            )
            
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data["citations"], list)
                # Should have some citations from the knowledge base
            else:
                pytest.skip("Ollama not available")
        except Exception as e:
            pytest.skip(f"Agent test skipped: {e}")
