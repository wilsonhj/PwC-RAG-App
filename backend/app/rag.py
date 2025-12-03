"""
RAG module using ChromaDB for vector storage and Ollama for embeddings + generation.

Now uses centralized configuration from config.py and modular pipeline components.

Multi-Model Strategy (mirrors Claude Haiku/Sonnet pattern):
- Router: Fast classification of query complexity
- Worker: Balanced quality for standard RAG queries
- Synthesis: High quality for complex reasoning

Light Setup (~2GB):
   ollama pull nomic-embed-text
   ollama pull qwen2.5:3b

Full Setup (~23GB, better quality):
   ollama pull nomic-embed-text
   ollama pull qwen2.5:3b
   ollama pull gemma3:12b
   ollama pull mistral-small

Prerequisites:
1. Install Ollama: https://ollama.com/download
2. Start server: ollama serve
3. Pull models (see above)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import time
import chromadb
import ollama

from .config import get_config, ModelTier, RetrievalStrategy
from .pipeline.chunker import RecursiveChunker, Chunk


# ---------------------------------------------------------------------------
# Get configuration
# ---------------------------------------------------------------------------
config = get_config()


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------
@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    model_tier: str
    model_used: str
    retrieval_strategy: str
    num_chunks_retrieved: int
    latency_embed_ms: float = 0.0
    latency_retrieve_ms: float = 0.0
    latency_generate_ms: float = 0.0
    latency_total_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def latency_breakdown(self) -> dict:
        return {
            "embed_ms": self.latency_embed_ms,
            "retrieve_ms": self.latency_retrieve_ms,
            "generate_ms": self.latency_generate_ms,
            "total_ms": self.latency_total_ms,
        }


# Store recent metrics for observability
_recent_metrics: List[QueryMetrics] = []
MAX_METRICS_HISTORY = 100

# Sample documents - replace with your own corpus
DOCUMENTS = [
    {
        "id": "chunk-1",
        "text": "PwC helps clients design, build, and deploy production-grade AI and LLM systems.",
    },
    {
        "id": "chunk-2",
        "text": "Retrieval-Augmented Generation (RAG) combines a retriever over private data with a generative model.",
    },
    {
        "id": "chunk-3",
        "text": "This proof-of-concept focuses on a simple question-answering experience over a small knowledge base.",
    },
    {
        "id": "chunk-4",
        "text": "PwC's AI consulting practice covers strategy, implementation, and managed services for enterprise clients.",
    },
    {
        "id": "chunk-5",
        "text": "Vector databases store embeddings and enable semantic similarity search over unstructured data.",
    },
]

# ---------------------------------------------------------------------------
# ChromaDB setup (persistent, uses config)
# ---------------------------------------------------------------------------
_client = chromadb.PersistentClient(path=config.vectordb.persist_path)
_collection = _client.get_or_create_collection(
    name=config.vectordb.collection_name,
    metadata={"hnsw:space": config.vectordb.distance_metric},
)


def _get_embedding(text: str) -> Tuple[List[float], float]:
    """Get embedding vector from Ollama with timing."""
    start = time.time()
    response = ollama.embeddings(model=config.embedding.model, prompt=text)
    latency_ms = (time.time() - start) * 1000
    return response["embedding"], latency_ms


def index_documents(documents: Optional[List[dict]] = None) -> int:
    """
    Index documents into ChromaDB. Safe to call multiple times.
    
    Args:
        documents: List of dicts with 'id' and 'text'. Uses DOCUMENTS if None.
        
    Returns:
        Number of documents indexed
    """
    docs = documents or DOCUMENTS
    existing_ids = set(_collection.get()["ids"])
    indexed = 0
    
    for doc in docs:
        if doc["id"] not in existing_ids:
            embedding, _ = _get_embedding(doc["text"])
            _collection.add(
                ids=[doc["id"]],
                embeddings=[embedding],
                documents=[doc["text"]],
                metadatas=[{"id": doc["id"], **doc.get("metadata", {})}],
            )
            indexed += 1
    
    total = _collection.count()
    print(f"Indexed {indexed} new documents. Total: {total} in ChromaDB.")
    return indexed


def retrieve(
    question: str,
    k: Optional[int] = None,
    strategy: Optional[RetrievalStrategy] = None,
) -> Tuple[List[dict], float]:
    """
    Retrieve relevant passages using configured strategy.
    
    Args:
        question: The query string
        k: Number of results (uses config default if None)
        strategy: Retrieval strategy (uses config default if None)
        
    Returns:
        Tuple of (passages list, latency in ms)
    """
    k = k or config.retrieval.top_k
    strategy = strategy or config.retrieval.strategy
    
    start = time.time()
    query_embedding, embed_latency = _get_embedding(question)
    
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    
    passages = []
    for i, doc_id in enumerate(results["ids"][0]):
        score = 1 - results["distances"][0][i]  # Convert distance to similarity
        
        # Apply similarity threshold
        if score >= config.retrieval.similarity_threshold:
            passages.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "score": score,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })
    
    total_latency = (time.time() - start) * 1000
    return passages, total_latency


def get_collection_stats() -> dict:
    """Get statistics about the vector collection."""
    return {
        "total_documents": _collection.count(),
        "collection_name": config.vectordb.collection_name,
        "distance_metric": config.vectordb.distance_metric,
    }


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are PwC's AI Briefing Assistant. Answer questions using ONLY the provided context.
- Cite sources using [chunk-id] format.
- If the context doesn't contain the answer, say "I don't have enough information to answer this."
- Be concise and professional."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

# Router prompt for classifying query complexity
ROUTER_PROMPT = """Classify this query into one of these categories:
- SIMPLE: Basic factual lookup, single-hop retrieval
- COMPLEX: Multi-step reasoning, comparison, synthesis needed

Query: {question}

Respond with only: SIMPLE or COMPLEX"""


def _call_llm(
    model_tier: ModelTier,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[str, float]:
    """
    Call Ollama with specified model tier.
    
    Returns:
        Tuple of (response content, latency in ms)
    """
    model = config.llm.get_model(model_tier)
    start = time.time()
    
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": config.llm.temperature,
            "num_predict": config.llm.max_tokens,
        },
    )
    
    latency_ms = (time.time() - start) * 1000
    return response["message"]["content"], latency_ms


def route_query(question: str) -> ModelTier:
    """Use router model to classify query complexity."""
    try:
        result, _ = _call_llm(
            ModelTier.ROUTER,
            "You are a query classifier. Respond with only SIMPLE or COMPLEX.",
            ROUTER_PROMPT.format(question=question),
        )
        # Parse response
        if "COMPLEX" in result.upper():
            return ModelTier.SYNTHESIS
        return ModelTier.WORKER
    except Exception as e:
        print(f"Router failed, defaulting to worker: {e}")
        return ModelTier.WORKER


def get_recent_metrics() -> List[QueryMetrics]:
    """Get recent query metrics for observability."""
    return _recent_metrics.copy()


def answer_question(
    question: str,
    use_routing: bool = True,
    top_k: Optional[int] = None,
    strategy: Optional[RetrievalStrategy] = None,
) -> Tuple[str, List[str], Optional[QueryMetrics]]:
    """
    Full RAG pipeline with optional multi-model routing and metrics.
    
    Args:
        question: The user's question
        use_routing: Whether to use the router to select model tier
        top_k: Override for number of chunks to retrieve
        strategy: Override for retrieval strategy
        
    Returns:
        Tuple of (answer, citations, metrics)
    """
    total_start = time.time()
    
    # Step 1: Route to appropriate model
    if use_routing:
        model_tier = route_query(question)
    else:
        model_tier = ModelTier.WORKER
    
    # Step 2: Retrieve with timing
    passages, retrieve_latency = retrieve(question, k=top_k, strategy=strategy)
    context = "\n".join(f"[{p['id']}] {p['text']}" for p in passages)
    citations = [p["id"] for p in passages]

    # Step 3: Generate with selected model
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    answer, generate_latency = _call_llm(model_tier, SYSTEM_PROMPT, user_prompt)
    
    total_latency = (time.time() - total_start) * 1000
    
    # Build metrics
    model_used = config.llm.get_model(model_tier)
    metrics = QueryMetrics(
        query=question,
        model_tier=model_tier.value,
        model_used=model_used,
        retrieval_strategy=(strategy or config.retrieval.strategy).value,
        num_chunks_retrieved=len(passages),
        latency_retrieve_ms=retrieve_latency,
        latency_generate_ms=generate_latency,
        latency_total_ms=total_latency,
    )
    
    # Store metrics
    _recent_metrics.append(metrics)
    if len(_recent_metrics) > MAX_METRICS_HISTORY:
        _recent_metrics.pop(0)
    
    # Add model info for debugging
    answer = f"[Model: {model_used}]\n\n{answer}"

    return answer, citations, metrics


# ---------------------------------------------------------------------------
# Auto-index on module load
# ---------------------------------------------------------------------------
try:
    index_documents()
except Exception as e:
    print(f"Warning: Could not index documents (Ollama may not be running): {e}")
