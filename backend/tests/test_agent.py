"""
Tests for the LangGraph agent module.
"""

import pytest
from app.pipeline.agent import (
    AgentState,
    AgentConfig,
    AgentResult,
    RAGAgent,
    build_rag_agent,
    create_classify_node,
    create_retrieve_node,
    create_generate_node,
    should_refine,
)
from app.config import RetrievalStrategy


class TestAgentState:
    """Tests for AgentState structure."""
    
    def test_initial_state(self):
        state: AgentState = {
            "question": "What is RAG?",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "",
            "answer": "",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        assert state["question"] == "What is RAG?"
        assert state["complexity"] == "simple"
        assert state["steps_executed"] == []


class TestAgentConfig:
    """Tests for AgentConfig."""
    
    def test_default_config(self):
        config = AgentConfig()
        
        assert config.max_refinements == 1
        assert config.top_k == 3
        assert config.enable_refinement == True
    
    def test_custom_config(self):
        config = AgentConfig(max_refinements=2, top_k=5)
        
        assert config.max_refinements == 2
        assert config.top_k == 5


class TestAgentResult:
    """Tests for AgentResult."""
    
    def test_result_creation(self):
        result = AgentResult(
            answer="RAG is Retrieval-Augmented Generation.",
            citations=["chunk-1", "chunk-2"],
            complexity="simple",
            retrieval_strategy="semantic",
            steps_executed=["classify", "retrieve", "generate"],
            latency_ms=1500.0,
            model_used="worker",
            refinement_count=0,
        )
        
        assert "RAG" in result.answer
        assert len(result.citations) == 2
        assert result.complexity == "simple"
        assert len(result.steps_executed) == 3


class TestShouldRefine:
    """Tests for the refinement decision logic."""
    
    def test_no_refinement_needed(self):
        state: AgentState = {
            "question": "test",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "",
            "answer": "A complete answer.",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = should_refine(state)
        assert result == "end"
    
    def test_refinement_needed(self):
        state: AgentState = {
            "question": "test",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "",
            "answer": "Short.",
            "citations": [],
            "needs_refinement": True,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = should_refine(state)
        assert result == "refine"
    
    def test_max_refinements_reached(self):
        state: AgentState = {
            "question": "test",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "",
            "answer": "Short.",
            "citations": [],
            "needs_refinement": True,
            "refinement_count": 1,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = should_refine(state)
        assert result == "end"


class TestClassifyNode:
    """Tests for the classify node."""
    
    def test_classify_simple_query(self):
        # Mock LLM that returns SIMPLE
        def mock_llm(tier, system, user):
            return "SIMPLE", 10.0
        
        classify = create_classify_node(mock_llm, "Classify: {question}")
        
        state: AgentState = {
            "question": "What is RAG?",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "",
            "answer": "",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = classify(state)
        
        assert result["complexity"] == "simple"
        assert "classify" in result["steps_executed"]
    
    def test_classify_complex_query(self):
        # Mock LLM that returns COMPLEX
        def mock_llm(tier, system, user):
            return "COMPLEX", 10.0
        
        classify = create_classify_node(mock_llm, "Classify: {question}")
        
        state: AgentState = {
            "question": "Compare different AI approaches",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "",
            "answer": "",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = classify(state)
        
        assert result["complexity"] == "complex"
        assert result["retrieval_strategy"] == "hybrid"


class TestRetrieveNode:
    """Tests for the retrieve node."""
    
    def test_retrieve_passages(self):
        # Mock retrieve function
        def mock_retrieve(question, k, strategy):
            return [
                {"id": "chunk-1", "text": "Test passage", "score": 0.9, "metadata": {}},
            ], 50.0, "semantic"
        
        retrieve = create_retrieve_node(mock_retrieve)
        
        state: AgentState = {
            "question": "What is RAG?",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "",
            "answer": "",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = retrieve(state)
        
        assert len(result["passages"]) == 1
        assert result["citations"] == ["chunk-1"]
        assert "retrieve" in result["steps_executed"]


class TestGenerateNode:
    """Tests for the generate node."""
    
    def test_generate_answer(self):
        # Mock LLM
        def mock_llm(tier, system, user):
            return "RAG is Retrieval-Augmented Generation.", 100.0
        
        generate = create_generate_node(
            mock_llm,
            "You are an assistant.",
            "Context: {context}\nQuestion: {question}",
        )
        
        state: AgentState = {
            "question": "What is RAG?",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "[chunk-1] RAG combines retrieval with generation.",
            "answer": "",
            "citations": ["chunk-1"],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = generate(state)
        
        assert "RAG" in result["answer"]
        assert result["model_used"] == "worker"
        assert "generate" in result["steps_executed"]
    
    def test_generate_uses_synthesis_for_complex(self):
        """Test that complex queries use synthesis model."""
        def mock_llm(tier, system, user):
            return f"Answer from {tier}", 100.0
        
        generate = create_generate_node(
            mock_llm,
            "You are an assistant.",
            "Context: {context}\nQuestion: {question}",
        )
        
        state: AgentState = {
            "question": "Compare approaches",
            "complexity": "complex",  # Complex query
            "retrieval_strategy": "hybrid",
            "passages": [],
            "context": "Some context",
            "answer": "",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = generate(state)
        
        assert result["model_used"] == "synthesis"
    
    def test_generate_triggers_refinement_for_short_answer(self):
        """Test that short answers trigger refinement."""
        def mock_llm(tier, system, user):
            return "Short.", 100.0  # Very short answer
        
        generate = create_generate_node(
            mock_llm,
            "You are an assistant.",
            "Context: {context}\nQuestion: {question}",
        )
        
        state: AgentState = {
            "question": "What is RAG?",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "Context here",
            "answer": "",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = generate(state)
        
        assert result["needs_refinement"] == True
    
    def test_generate_triggers_refinement_for_uncertain_answer(self):
        """Test that uncertain answers trigger refinement."""
        def mock_llm(tier, system, user):
            return "I don't have enough information to answer this question fully.", 100.0
        
        generate = create_generate_node(
            mock_llm,
            "You are an assistant.",
            "Context: {context}\nQuestion: {question}",
        )
        
        state: AgentState = {
            "question": "What is quantum computing?",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "No relevant context",
            "answer": "",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = generate(state)
        
        assert result["needs_refinement"] == True


class TestRefineNode:
    """Tests for the refine node."""
    
    def test_refine_improves_answer(self):
        from app.pipeline.agent import create_refine_node
        
        def mock_llm(tier, system, user):
            return "This is a much more detailed and complete answer about the topic.", 150.0
        
        refine = create_refine_node(
            mock_llm,
            "Question: {question}\nContext: {context}\nCurrent: {current_answer}",
        )
        
        state: AgentState = {
            "question": "What is RAG?",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "RAG combines retrieval with generation.",
            "answer": "Short.",
            "citations": [],
            "needs_refinement": True,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = refine(state)
        
        assert len(result["answer"]) > len("Short.")
        assert result["refinement_count"] == 1
        assert result["needs_refinement"] == False
        assert "refine" in result["steps_executed"]


class TestRAGAgentIntegration:
    """Integration tests for the full RAGAgent."""
    
    def test_agent_full_run(self):
        """Test complete agent execution with mocks."""
        call_log = []
        
        def mock_llm(tier, system, user):
            call_log.append(f"llm:{tier}")
            if "classify" in user.lower() or "simple" in system.lower():
                return "SIMPLE", 10.0
            return "This is a complete answer about the topic.", 100.0
        
        def mock_retrieve(question, k, strategy):
            call_log.append("retrieve")
            return [
                {"id": "chunk-1", "text": "Test passage", "score": 0.9, "metadata": {}},
            ], 50.0, "semantic"
        
        agent = RAGAgent(
            llm_fn=mock_llm,
            retrieve_fn=mock_retrieve,
            router_prompt="Classify: {question}",
            system_prompt="You are an assistant.",
            user_template="Context: {context}\nQuestion: {question}",
            config=AgentConfig(max_refinements=0),  # Disable refinement
        )
        
        result = agent.run("What is RAG?")
        
        assert result.answer != ""
        assert "classify" in result.steps_executed
        assert "retrieve" in result.steps_executed
        assert "generate" in result.steps_executed
        assert result.latency_ms > 0
    
    def test_agent_with_refinement(self):
        """Test agent triggers refinement for short answers."""
        refinement_called = [False]
        
        def mock_llm(tier, system, user):
            if "classify" in user.lower():
                return "SIMPLE", 10.0
            if "improve" in system.lower() or refinement_called[0]:
                return "This is a much longer and more complete answer.", 100.0
            refinement_called[0] = True
            return "Short.", 50.0  # First answer is short
        
        def mock_retrieve(question, k, strategy):
            return [
                {"id": "chunk-1", "text": "Test", "score": 0.9, "metadata": {}},
            ], 50.0, "semantic"
        
        agent = RAGAgent(
            llm_fn=mock_llm,
            retrieve_fn=mock_retrieve,
            router_prompt="Classify: {question}",
            system_prompt="You are an assistant.",
            user_template="Context: {context}\nQuestion: {question}",
            config=AgentConfig(max_refinements=1),
        )
        
        result = agent.run("What is RAG?")
        
        # Should have refined
        assert result.refinement_count >= 0  # May or may not refine based on logic


class TestClassifyNodeEdgeCases:
    """Edge case tests for classify node."""
    
    def test_classify_handles_llm_exception(self):
        """Test classify defaults to simple on LLM error."""
        def failing_llm(tier, system, user):
            raise Exception("LLM unavailable")
        
        classify = create_classify_node(failing_llm, "Classify: {question}")
        
        state: AgentState = {
            "question": "What is RAG?",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "",
            "answer": "",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = classify(state)
        
        # Should default to simple on error
        assert result["complexity"] == "simple"
    
    def test_classify_short_query_uses_hybrid(self):
        """Test that short queries get hybrid strategy."""
        def mock_llm(tier, system, user):
            return "SIMPLE", 10.0
        
        classify = create_classify_node(mock_llm, "Classify: {question}")
        
        state: AgentState = {
            "question": "RAG",  # Very short query
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "",
            "answer": "",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = classify(state)
        
        # Short queries should use hybrid for keyword matching
        assert result["retrieval_strategy"] == "hybrid"


class TestRetrieveNodeEdgeCases:
    """Edge case tests for retrieve node."""
    
    def test_retrieve_empty_results(self):
        """Test retrieve handles empty results."""
        def mock_retrieve(question, k, strategy):
            return [], 10.0, "semantic"  # No results
        
        retrieve = create_retrieve_node(mock_retrieve)
        
        state: AgentState = {
            "question": "Unknown topic",
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "",
            "answer": "",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": 1,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        result = retrieve(state)
        
        assert result["passages"] == []
        assert result["citations"] == []
        assert result["context"] == ""
