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
from app.pipeline.tokenizer import TokenUsage
from app.config import RetrievalStrategy


def make_state(**overrides) -> AgentState:
    base: AgentState = {
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
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    base.update(overrides)
    return base


class TestAgentState:
    """Tests for AgentState structure."""
    
    def test_initial_state(self):
        state = make_state()
        assert state["question"] == "What is RAG?"
        assert state["complexity"] == "simple"
        assert state["steps_executed"] == []


class TestAgentConfig:
    """Tests for AgentConfig."""
    
    def test_default_config(self):
        config = AgentConfig()
        
        assert config.max_refinements == 1
        assert config.top_k == 3
        assert config.enable_refinement is True
    
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
            input_tokens=42,
            output_tokens=18,
            total_tokens=60,
        )
        
        assert "RAG" in result.answer
        assert len(result.citations) == 2
        assert result.total_tokens == 60


class TestShouldRefine:
    """Tests for the refinement decision logic."""
    
    def test_no_refinement_needed(self):
        state = make_state(answer="A complete answer.")
        assert should_refine(state) == "end"
    
    def test_refinement_needed(self):
        state = make_state(answer="Short.", needs_refinement=True)
        assert should_refine(state) == "refine"
    
    def test_max_refinements_reached(self):
        state = make_state(answer="Short.", needs_refinement=True, refinement_count=1)
        assert should_refine(state) == "end"


class TestClassifyNode:
    """Tests for the classify node."""
    
    def test_classify_simple_query(self):
        classify = create_classify_node(lambda *args: ("SIMPLE", 10.0, None), "Classify: {question}")
        result = classify(make_state())
        assert result["complexity"] == "simple"
        assert "classify" in result["steps_executed"]
    
    def test_classify_complex_query(self):
        classify = create_classify_node(lambda *args: ("COMPLEX", 10.0, None), "Classify: {question}")
        result = classify(make_state(question="Compare different AI approaches"))
        assert result["complexity"] == "complex"
        assert result["retrieval_strategy"] == "hybrid"


class TestRetrieveNode:
    """Tests for the retrieve node."""
    
    def test_retrieve_passages(self):
        def mock_retrieve(question, k, strategy):
            return [
                {"id": "chunk-1", "text": "Test passage", "score": 0.9, "metadata": {}},
            ], 50.0, "semantic"
        retrieve = create_retrieve_node(mock_retrieve)
        result = retrieve(make_state())
        assert len(result["passages"]) == 1
        assert result["citations"] == ["chunk-1"]
        assert "retrieve" in result["steps_executed"]


class TestGenerateNode:
    """Tests for the generate node."""
    
    def test_generate_answer(self):
        def mock_llm(tier, system, user):
            return "RAG is Retrieval-Augmented Generation.", 100.0, TokenUsage(10, 5, 15)
        generate = create_generate_node(mock_llm, "You are an assistant.", "Context: {context}\nQuestion: {question}")
        result = generate(make_state(context="[chunk-1] RAG combines retrieval with generation.", citations=["chunk-1"]))
        assert result["model_used"] == "worker"
        assert result["total_tokens"] == 15
    
    def test_generate_uses_synthesis_for_complex(self):
        def mock_llm(tier, system, user):
            return f"Answer from {tier}", 100.0, TokenUsage(8, 4, 12)
        generate = create_generate_node(mock_llm, "You are an assistant.", "Context: {context}\nQuestion: {question}")
        result = generate(make_state(question="Compare approaches", complexity="complex", retrieval_strategy="hybrid"))
        assert result["model_used"] == "synthesis"
        assert result["total_tokens"] == 12
    
    def test_generate_triggers_refinement_for_short_answer(self):
        def mock_llm(tier, system, user):
            return "Short.", 100.0, TokenUsage(5, 2, 7)
        generate = create_generate_node(mock_llm, "You are an assistant.", "Context: {context}\nQuestion: {question}")
        result = generate(make_state(context="Context here"))
        assert result["needs_refinement"] is True
        assert result["total_tokens"] == 7

    def test_generate_triggers_refinement_for_uncertain_answer(self):
        def mock_llm(tier, system, user):
            return "I don't have enough information to answer this question fully.", 100.0, TokenUsage(6, 3, 9)
        generate = create_generate_node(mock_llm, "You are an assistant.", "Context: {context}\nQuestion: {question}")
        result = generate(make_state(question="What is quantum computing?", context="No relevant context"))
        assert result["needs_refinement"] is True
        assert result["total_tokens"] == 9


class TestRefineNode:
    """Tests for the refine node."""
    
    def test_refine_improves_answer(self):
        from app.pipeline.agent import create_refine_node

        def mock_llm(tier, system, user):
            return "This is a much more detailed and complete answer about the topic.", 150.0, TokenUsage(7, 6, 13)

        refine = create_refine_node(mock_llm, "Question: {question}\nContext: {context}\nCurrent: {current_answer}")
        result = refine(make_state(context="RAG combines retrieval with generation.", answer="Short.", needs_refinement=True))
        assert len(result["answer"]) > len("Short.")
        assert result["refinement_count"] == 1
        assert result["total_tokens"] == 13


class TestRAGAgentIntegration:
    """Integration tests for the full RAGAgent."""
    
    def test_agent_full_run(self):
        call_log = []

        def mock_llm(tier, system, user):
            call_log.append(f"llm:{tier}")
            if "classify" in user.lower() or "simple" in system.lower():
                return "SIMPLE", 10.0, None
            return "This is a complete answer about the topic.", 100.0, TokenUsage(20, 10, 30)

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
            config=AgentConfig(max_refinements=0),
        )

        result = agent.run("What is RAG?")
        assert result.total_tokens == 30

    def test_agent_with_refinement(self):
        refinement_called = [False]

        def mock_llm(tier, system, user):
            if "classify" in user.lower():
                return "SIMPLE", 10.0, None
            if "improve" in system.lower() or refinement_called[0]:
                return "This is a much longer and more complete answer.", 100.0, TokenUsage(15, 20, 35)
            refinement_called[0] = True
            return "Short.", 50.0, TokenUsage(10, 5, 15)

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
        assert result.total_tokens >= 15


class TestClassifyNodeEdgeCases:
    """Edge case tests for classify node."""
    
    def test_classify_handles_llm_exception(self):
        classify = create_classify_node(lambda *args: (_ for _ in ()).throw(Exception("LLM unavailable")), "Classify: {question}")
        result = classify(make_state())
        assert result["complexity"] == "simple"

    def test_classify_short_query_uses_hybrid(self):
        classify = create_classify_node(lambda *args: ("SIMPLE", 10.0, None), "Classify: {question}")
        result = classify(make_state(question="RAG"))
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
