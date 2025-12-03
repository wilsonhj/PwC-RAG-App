"""
RAG module using ChromaDB for vector storage and Ollama for embeddings + generation.

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

from typing import List, Tuple
import chromadb
import ollama

# ---------------------------------------------------------------------------
# Configuration - Multi-Model Strategy
# ---------------------------------------------------------------------------
# Mirrors Claude Haiku/Sonnet pattern:
# - Router: cheap/fast for classification and routing decisions
# - Worker: balanced for retrieval QA and standard tasks  
# - Synthesis: high quality for complex reasoning and final answers

EMBEDDING_MODEL = "nomic-embed-text"

# Model tiers (pull these with: ollama pull <model>)
# Light setup: uses qwen2.5:3b for everything (~2GB total)
# Full setup: uncomment larger models for better quality
MODELS = {
    "router": "qwen2.5:3b",        # Fast routing/classification (~2GB)
    "worker": "qwen2.5:3b",        # Light: same as router | Full: gemma3:12b
    "synthesis": "qwen2.5:3b",     # Light: same as router | Full: mistral-small
}

# Default model for simple queries
DEFAULT_MODEL = "worker"

# To upgrade to full multi-model setup, uncomment:
# MODELS = {
#     "router": "qwen2.5:3b",        # Fast routing/classification (~2GB)
#     "worker": "gemma3:12b",        # Balanced quality for RAG QA (~7GB)
#     "synthesis": "mistral-small",  # Best quality for complex tasks (~14GB)
# }

COLLECTION_NAME = "pwc_docs"

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
# ChromaDB setup (persistent, stored in ./chroma_db)
# ---------------------------------------------------------------------------
_client = chromadb.PersistentClient(path="./chroma_db")
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama."""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


def index_documents() -> None:
    """Index all documents into ChromaDB. Safe to call multiple times."""
    existing_ids = set(_collection.get()["ids"])
    for doc in DOCUMENTS:
        if doc["id"] not in existing_ids:
            embedding = _get_embedding(doc["text"])
            _collection.add(
                ids=[doc["id"]],
                embeddings=[embedding],
                documents=[doc["text"]],
                metadatas=[{"id": doc["id"]}],
            )
    print(f"Indexed {_collection.count()} documents in ChromaDB.")


def retrieve(question: str, k: int = 3) -> List[dict]:
    """Semantic retrieval using ChromaDB."""
    query_embedding = _get_embedding(question)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    for i, doc_id in enumerate(results["ids"][0]):
        passages.append({
            "id": doc_id,
            "text": results["documents"][0][i],
            "score": 1 - results["distances"][0][i],  # Convert distance to similarity
        })
    return passages


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


def _call_llm(model_tier: str, system_prompt: str, user_prompt: str) -> str:
    """Call Ollama with specified model tier."""
    model = MODELS.get(model_tier, MODELS[DEFAULT_MODEL])
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


def route_query(question: str) -> str:
    """Use router model to classify query complexity."""
    try:
        result = _call_llm(
            "router",
            "You are a query classifier. Respond with only SIMPLE or COMPLEX.",
            ROUTER_PROMPT.format(question=question),
        )
        # Parse response
        if "COMPLEX" in result.upper():
            return "synthesis"
        return "worker"
    except Exception as e:
        print(f"Router failed, defaulting to worker: {e}")
        return "worker"


def answer_question(question: str, use_routing: bool = True) -> Tuple[str, List[str]]:
    """
    Full RAG pipeline with optional multi-model routing:
    1. (Optional) Route query to appropriate model tier
    2. Retrieve relevant passages from ChromaDB
    3. Build prompt with context
    4. Generate answer using selected Ollama model
    """
    # Step 1: Route to appropriate model
    if use_routing:
        model_tier = route_query(question)
    else:
        model_tier = DEFAULT_MODEL
    
    # Step 2: Retrieve
    passages = retrieve(question, k=3)
    context = "\n".join(f"[{p['id']}] {p['text']}" for p in passages)
    citations = [p["id"] for p in passages]

    # Step 3: Generate with selected model
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    answer = _call_llm(model_tier, SYSTEM_PROMPT, user_prompt)
    
    # Add model info for debugging
    model_used = MODELS.get(model_tier, MODELS[DEFAULT_MODEL])
    answer = f"[Model: {model_used}]\n\n{answer}"

    return answer, citations


# ---------------------------------------------------------------------------
# Auto-index on module load
# ---------------------------------------------------------------------------
try:
    index_documents()
except Exception as e:
    print(f"Warning: Could not index documents (Ollama may not be running): {e}")
