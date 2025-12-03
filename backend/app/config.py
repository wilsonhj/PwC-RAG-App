"""
Centralized configuration for PWC RAG App.

All configurable parameters in one place for easy tuning and environment-based overrides.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import os


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    SEMANTIC = "semantic"      # Pure vector similarity
    HYBRID = "hybrid"          # Vector + BM25 keyword search
    ADAPTIVE = "adaptive"      # Router selects based on query


class ModelTier(str, Enum):
    """Model tiers for multi-model routing."""
    ROUTER = "router"          # Fast classification
    WORKER = "worker"          # Standard RAG tasks
    SYNTHESIS = "synthesis"    # Complex reasoning


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model: str = "nomic-embed-text"
    dimension: int = 768  # nomic-embed-text dimension


@dataclass
class LLMConfig:
    """LLM configuration with multi-model tiers."""
    # Model assignments per tier
    models: dict = field(default_factory=lambda: {
        ModelTier.ROUTER: "qwen2.5:3b",
        ModelTier.WORKER: "qwen2.5:3b",
        ModelTier.SYNTHESIS: "qwen2.5:3b",
    })
    default_tier: ModelTier = ModelTier.WORKER
    
    # Generation parameters
    temperature: float = 0.1
    max_tokens: int = 1024
    
    def get_model(self, tier: ModelTier) -> str:
        """Get model name for a given tier."""
        return self.models.get(tier, self.models[self.default_tier])


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    chunk_size: int = 512          # Characters per chunk
    chunk_overlap: int = 50        # Overlap between chunks
    min_chunk_size: int = 100      # Minimum chunk size to keep
    
    # Separators for recursive splitting (in order of priority)
    separators: list = field(default_factory=lambda: [
        "\n\n",      # Paragraphs
        "\n",        # Lines
        ". ",        # Sentences
        ", ",        # Clauses
        " ",         # Words
        "",          # Characters
    ])


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC
    top_k: int = 3                          # Number of chunks to retrieve
    similarity_threshold: float = 0.5       # Minimum similarity score
    
    # Hybrid search weights (semantic vs keyword)
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    
    # Adaptive retrieval thresholds
    complexity_threshold: float = 0.6       # Above this = complex query


@dataclass
class VectorDBConfig:
    """Vector database configuration."""
    provider: str = "chromadb"
    persist_path: str = "./chroma_db"
    collection_name: str = "pwc_docs"
    distance_metric: str = "cosine"         # cosine, l2, ip


@dataclass
class ObservabilityConfig:
    """Observability and metrics configuration."""
    enable_token_tracking: bool = True
    enable_latency_tracking: bool = True
    log_level: str = "INFO"
    
    # Token counting model (for accurate counts)
    tokenizer_model: str = "cl100k_base"    # OpenAI's tokenizer


@dataclass
class AppConfig:
    """Main application configuration."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    vectordb: VectorDBConfig = field(default_factory=VectorDBConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list = field(default_factory=lambda: [
        "http://localhost:5173",
        "http://localhost:3000",
        "*",
    ])


# ---------------------------------------------------------------------------
# Global config instance (singleton pattern)
# ---------------------------------------------------------------------------
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def update_config(**kwargs) -> AppConfig:
    """Update configuration with new values."""
    global _config
    config = get_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# ---------------------------------------------------------------------------
# Environment-based overrides
# ---------------------------------------------------------------------------
def load_from_env() -> AppConfig:
    """Load configuration from environment variables."""
    config = get_config()
    
    # Override from environment
    if os.getenv("EMBEDDING_MODEL"):
        config.embedding.model = os.getenv("EMBEDDING_MODEL")
    
    if os.getenv("LLM_MODEL"):
        # Set all tiers to the same model
        model = os.getenv("LLM_MODEL")
        config.llm.models = {
            ModelTier.ROUTER: model,
            ModelTier.WORKER: model,
            ModelTier.SYNTHESIS: model,
        }
    
    if os.getenv("RETRIEVAL_STRATEGY"):
        config.retrieval.strategy = RetrievalStrategy(os.getenv("RETRIEVAL_STRATEGY"))
    
    if os.getenv("TOP_K"):
        config.retrieval.top_k = int(os.getenv("TOP_K"))
    
    if os.getenv("CHUNK_SIZE"):
        config.chunking.chunk_size = int(os.getenv("CHUNK_SIZE"))
    
    if os.getenv("VECTORDB_PATH"):
        config.vectordb.persist_path = os.getenv("VECTORDB_PATH")
    
    return config
