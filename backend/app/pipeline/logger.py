"""
Structured logging for the RAG pipeline.

Provides JSON-formatted logs for:
- Query processing
- Retrieval operations
- LLM calls
- Error tracking

Designed for production observability and log aggregation.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Optional, Dict
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    event: str
    component: str
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    trace_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class StructuredLogger:
    """
    Structured JSON logger for the RAG pipeline.
    
    Outputs logs in JSON format for easy parsing by log aggregators
    like ELK, Datadog, or CloudWatch.
    """
    
    def __init__(
        self,
        name: str = "pwc-rag",
        level: LogLevel = LogLevel.INFO,
        output_json: bool = True,
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Minimum log level
            output_json: Whether to output JSON format
        """
        self.name = name
        self.level = level
        self.output_json = output_json
        self._trace_id: Optional[str] = None
        
        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Add stream handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.value))
        
        if output_json:
            handler.setFormatter(logging.Formatter('%(message)s'))
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        self.logger.addHandler(handler)
    
    def set_trace_id(self, trace_id: str):
        """Set trace ID for request correlation."""
        self._trace_id = trace_id
    
    def clear_trace_id(self):
        """Clear trace ID."""
        self._trace_id = None
    
    def _log(
        self,
        level: LogLevel,
        event: str,
        component: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """Internal logging method."""
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=level.value,
            event=event,
            component=component,
            message=message,
            data=data,
            error=error,
            trace_id=self._trace_id,
        )
        
        if self.output_json:
            log_msg = entry.to_json()
        else:
            log_msg = f"[{event}] {component}: {message}"
            if data:
                log_msg += f" | {data}"
        
        log_method = getattr(self.logger, level.value.lower())
        log_method(log_msg)
    
    def debug(
        self,
        event: str,
        component: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Log debug message."""
        self._log(LogLevel.DEBUG, event, component, message, data)
    
    def info(
        self,
        event: str,
        component: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Log info message."""
        self._log(LogLevel.INFO, event, component, message, data)
    
    def warning(
        self,
        event: str,
        component: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Log warning message."""
        self._log(LogLevel.WARNING, event, component, message, data)
    
    def error(
        self,
        event: str,
        component: str,
        message: str,
        error: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Log error message."""
        self._log(LogLevel.ERROR, event, component, message, data, error)
    
    # ---------------------------------------------------------------------------
    # Convenience methods for common events
    # ---------------------------------------------------------------------------
    def log_query_start(self, question: str, strategy: str):
        """Log query start."""
        self.info(
            event="query_start",
            component="rag",
            message=f"Processing query: {question[:50]}...",
            data={"question_length": len(question), "strategy": strategy},
        )
    
    def log_query_complete(
        self,
        question: str,
        latency_ms: float,
        num_chunks: int,
        model_used: str,
    ):
        """Log query completion."""
        self.info(
            event="query_complete",
            component="rag",
            message="Query processed successfully",
            data={
                "latency_ms": round(latency_ms, 2),
                "num_chunks": num_chunks,
                "model_used": model_used,
            },
        )
    
    def log_retrieval(
        self,
        strategy: str,
        num_results: int,
        latency_ms: float,
    ):
        """Log retrieval operation."""
        self.debug(
            event="retrieval",
            component="retriever",
            message=f"Retrieved {num_results} passages",
            data={
                "strategy": strategy,
                "num_results": num_results,
                "latency_ms": round(latency_ms, 2),
            },
        )
    
    def log_llm_call(
        self,
        model: str,
        tier: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ):
        """Log LLM call."""
        self.debug(
            event="llm_call",
            component="llm",
            message=f"LLM call to {model}",
            data={
                "model": model,
                "tier": tier,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": round(latency_ms, 2),
            },
        )
    
    def log_ingestion(
        self,
        source_id: str,
        chunks_created: int,
        chunks_indexed: int,
    ):
        """Log document ingestion."""
        self.info(
            event="ingestion",
            component="ingest",
            message=f"Ingested document {source_id}",
            data={
                "source_id": source_id,
                "chunks_created": chunks_created,
                "chunks_indexed": chunks_indexed,
            },
        )
    
    def log_error(self, component: str, error: Exception, context: Optional[dict] = None):
        """Log an error with context."""
        self.error(
            event="error",
            component=component,
            message=str(error),
            error=type(error).__name__,
            data=context,
        )


# ---------------------------------------------------------------------------
# Global instance
# ---------------------------------------------------------------------------
_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Get or create the global structured logger."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger()
    return _logger


def configure_logger(
    level: LogLevel = LogLevel.INFO,
    output_json: bool = True,
) -> StructuredLogger:
    """Configure and return the global logger."""
    global _logger
    _logger = StructuredLogger(level=level, output_json=output_json)
    return _logger
