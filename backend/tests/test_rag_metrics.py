"""Tests covering RAG metrics, logging, and routing fallbacks."""

import pytest

from app import rag
from app.config import ModelTier
from app.pipeline.tokenizer import TokenUsage


class StubLogger:
    """Simple logger stub to capture events for assertions."""

    def __init__(self):
        self.query_start = None
        self.retrieval = None
        self.query_complete = None
        self.errors = []

    def log_query_start(self, question, strategy):
        self.query_start = (question, strategy)

    def log_retrieval(self, strategy, num_results, latency_ms):
        self.retrieval = (strategy, num_results, latency_ms)

    def log_query_complete(self, question, latency_ms, num_chunks, model_used):
        self.query_complete = (question, latency_ms, num_chunks, model_used)

    def log_llm_call(self, *args, **kwargs):
        # Not needed for these tests but method must exist
        pass

    def log_error(self, component, error, context=None):
        self.errors.append((component, str(error), context))


@pytest.fixture(autouse=True)
def reset_metrics():
    """Ensure recent metrics buffer is cleared between tests."""
    rag._recent_metrics.clear()
    yield
    rag._recent_metrics.clear()


def test_answer_question_tracks_tokens_and_logs(monkeypatch):
    stub_logger = StubLogger()
    monkeypatch.setattr(rag, "get_logger", lambda: stub_logger)
    monkeypatch.setattr(rag, "route_query", lambda q: ModelTier.WORKER)

    def fake_retrieve(question, k=None, strategy=None):
        return (
            [
                {
                    "id": "chunk-1",
                    "text": "Context chunk text",
                    "score": 0.99,
                    "metadata": {},
                }
            ],
            12.34,
            "semantic",
        )

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)

    def fake_call_llm(model_tier, system_prompt, user_prompt, track_tokens=True):
        return "Answer text", 56.78, TokenUsage(input_tokens=120, output_tokens=48, total_tokens=168)

    monkeypatch.setattr(rag, "_call_llm", fake_call_llm)

    answer, citations, metrics = rag.answer_question("What is RAG?", use_routing=False)

    assert "Answer text" in answer
    assert citations == ["chunk-1"]
    assert metrics.input_tokens == 120
    assert metrics.output_tokens == 48
    assert metrics.latency_retrieve_ms == 12.34
    assert stub_logger.query_start[0] == "What is RAG?"
    assert stub_logger.retrieval[1] == 1
    assert stub_logger.query_complete[2] == 1  # num_chunks


def test_route_query_fallback_logs_error(monkeypatch):
    stub_logger = StubLogger()
    monkeypatch.setattr(rag, "get_logger", lambda: stub_logger)

    def failing_call_llm(*args, **kwargs):
        raise RuntimeError("router down")

    monkeypatch.setattr(rag, "_call_llm", failing_call_llm)

    result = rag.route_query("Test question")

    assert result == ModelTier.WORKER
    assert stub_logger.errors
    component, message, _ = stub_logger.errors[0]
    assert component == "router"
    assert "router down" in message
