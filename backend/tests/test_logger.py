"""
Tests for the structured logger module.
"""

import pytest
import json
from app.pipeline.logger import (
    StructuredLogger,
    LogLevel,
    LogEntry,
    get_logger,
    configure_logger,
)


class TestLogEntry:
    """Tests for LogEntry dataclass."""
    
    def test_log_entry_creation(self):
        entry = LogEntry(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            event="test_event",
            component="test",
            message="Test message",
        )
        
        assert entry.level == "INFO"
        assert entry.event == "test_event"
        assert entry.component == "test"
    
    def test_log_entry_to_dict(self):
        entry = LogEntry(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            event="test",
            component="test",
            message="Test",
            data={"key": "value"},
        )
        
        d = entry.to_dict()
        
        assert "timestamp" in d
        assert "level" in d
        assert "data" in d
        assert d["data"]["key"] == "value"
    
    def test_log_entry_excludes_none(self):
        entry = LogEntry(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            event="test",
            component="test",
            message="Test",
            data=None,
            error=None,
        )
        
        d = entry.to_dict()
        
        assert "data" not in d
        assert "error" not in d
    
    def test_log_entry_to_json(self):
        entry = LogEntry(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            event="test",
            component="test",
            message="Test",
        )
        
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["level"] == "INFO"
        assert parsed["event"] == "test"


class TestStructuredLogger:
    """Tests for StructuredLogger."""
    
    def test_logger_creation(self):
        logger = StructuredLogger(name="test-logger")
        
        assert logger.name == "test-logger"
        assert logger.level == LogLevel.INFO
    
    def test_logger_with_custom_level(self):
        logger = StructuredLogger(level=LogLevel.DEBUG)
        
        assert logger.level == LogLevel.DEBUG
    
    def test_set_trace_id(self):
        logger = StructuredLogger()
        logger.set_trace_id("trace-123")
        
        assert logger._trace_id == "trace-123"
        
        logger.clear_trace_id()
        assert logger._trace_id is None
    
    def test_log_query_start(self, capsys):
        logger = StructuredLogger(output_json=True)
        logger.log_query_start("What is RAG?", "semantic")
        
        captured = capsys.readouterr()
        log_data = json.loads(captured.out.strip())
        
        assert log_data["event"] == "query_start"
        assert log_data["component"] == "rag"
        assert "question_length" in log_data["data"]
    
    def test_log_query_complete(self, capsys):
        logger = StructuredLogger(output_json=True)
        logger.log_query_complete(
            question="What is RAG?",
            latency_ms=1500.0,
            num_chunks=3,
            model_used="qwen2.5:3b",
        )
        
        captured = capsys.readouterr()
        log_data = json.loads(captured.out.strip())
        
        assert log_data["event"] == "query_complete"
        assert log_data["data"]["latency_ms"] == 1500.0
        assert log_data["data"]["num_chunks"] == 3
    
    def test_log_llm_call(self, capsys):
        logger = StructuredLogger(output_json=True, level=LogLevel.DEBUG)
        logger.log_llm_call(
            model="qwen2.5:3b",
            tier="worker",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500.0,
        )
        
        captured = capsys.readouterr()
        log_data = json.loads(captured.out.strip())
        
        assert log_data["event"] == "llm_call"
        assert log_data["data"]["input_tokens"] == 100
        assert log_data["data"]["output_tokens"] == 50
    
    def test_log_error(self, capsys):
        logger = StructuredLogger(output_json=True)
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.log_error("test", e, {"context": "testing"})
        
        captured = capsys.readouterr()
        log_data = json.loads(captured.out.strip())
        
        assert log_data["event"] == "error"
        assert log_data["error"] == "ValueError"
        assert "Test error" in log_data["message"]


class TestLoggerSingleton:
    """Tests for global logger functions."""
    
    def test_get_logger_singleton(self):
        logger1 = get_logger()
        logger2 = get_logger()
        
        # Should return same instance
        assert logger1 is logger2
    
    def test_configure_logger(self):
        logger = configure_logger(level=LogLevel.DEBUG, output_json=False)
        
        assert logger.level == LogLevel.DEBUG
        assert logger.output_json == False
