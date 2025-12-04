"""
LangGraph-based agentic RAG workflow.

Implements a state machine with nodes:
1. classify: Determine query complexity and retrieval strategy
2. retrieve: Fetch relevant documents
3. generate: Produce answer from context
4. refine (optional): Improve answer if needed

The graph enables multi-hop reasoning and adaptive behavior.
"""

from typing import TypedDict, Literal, Optional, List, Annotated
from dataclasses import dataclass, field
import time
import operator

from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """State passed between nodes in the agent graph."""
    # Input
    question: str
    
    # Classification
    complexity: Literal["simple", "complex"]
    retrieval_strategy: Literal["semantic", "hybrid", "adaptive"]
    
    # Retrieval
    passages: List[dict]
    context: str
    
    # Generation
    answer: str
    citations: List[str]
    
    # Control flow
    needs_refinement: bool
    refinement_count: int
    max_refinements: int
    
    # Metrics
    steps_executed: Annotated[List[str], operator.add]
    latency_ms: float
    model_used: str


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_refinements: int = 1
    top_k: int = 3
    complexity_threshold: float = 0.6
    enable_refinement: bool = True


@dataclass
class AgentResult:
    """Result from agent execution."""
    answer: str
    citations: List[str]
    complexity: str
    retrieval_strategy: str
    steps_executed: List[str]
    latency_ms: float
    model_used: str
    refinement_count: int = 0


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------
def create_classify_node(llm_fn, router_prompt: str):
    """
    Create the classification node.
    
    Determines query complexity and optimal retrieval strategy.
    """
    def classify(state: AgentState) -> AgentState:
        question = state["question"]
        
        # Use LLM to classify
        try:
            result, _ = llm_fn(
                "router",
                "You are a query classifier. Respond with SIMPLE or COMPLEX.",
                router_prompt.format(question=question),
            )
            
            complexity = "complex" if "COMPLEX" in result.upper() else "simple"
        except Exception:
            complexity = "simple"
        
        # Determine retrieval strategy based on complexity
        if complexity == "complex":
            strategy = "hybrid"
        else:
            # Use heuristics for simple queries
            words = question.lower().split()
            if len(words) <= 3:
                strategy = "hybrid"  # Short queries benefit from keyword matching
            else:
                strategy = "semantic"
        
        return {
            **state,
            "complexity": complexity,
            "retrieval_strategy": strategy,
            "steps_executed": ["classify"],
        }
    
    return classify


def create_retrieve_node(retrieve_fn):
    """
    Create the retrieval node.
    
    Fetches relevant passages using the determined strategy.
    """
    def retrieve(state: AgentState) -> AgentState:
        question = state["question"]
        strategy = state.get("retrieval_strategy", "semantic")
        
        # Map string to enum
        from ..config import RetrievalStrategy
        strategy_enum = RetrievalStrategy(strategy)
        
        # Retrieve passages
        passages, latency, strategy_used = retrieve_fn(
            question=question,
            k=3,
            strategy=strategy_enum,
        )
        
        # Build context string
        context = "\n".join(
            f"[{p['id']}] {p['text']}" for p in passages
        )
        citations = [p["id"] for p in passages]
        
        return {
            **state,
            "passages": passages,
            "context": context,
            "citations": citations,
            "steps_executed": ["retrieve"],
        }
    
    return retrieve


def create_generate_node(llm_fn, system_prompt: str, user_template: str):
    """
    Create the generation node.
    
    Produces an answer from the retrieved context.
    """
    def generate(state: AgentState) -> AgentState:
        question = state["question"]
        context = state.get("context", "")
        complexity = state.get("complexity", "simple")
        
        # Select model tier based on complexity
        model_tier = "synthesis" if complexity == "complex" else "worker"
        
        # Generate answer
        user_prompt = user_template.format(context=context, question=question)
        answer, latency = llm_fn(model_tier, system_prompt, user_prompt)
        
        # Check if answer needs refinement
        needs_refinement = False
        if state.get("max_refinements", 1) > state.get("refinement_count", 0):
            # Simple heuristic: refine if answer is too short or uncertain
            if len(answer) < 50 or "don't have enough" in answer.lower():
                needs_refinement = True
        
        return {
            **state,
            "answer": answer,
            "needs_refinement": needs_refinement,
            "model_used": model_tier,
            "steps_executed": ["generate"],
        }
    
    return generate


def create_refine_node(llm_fn, refine_prompt: str):
    """
    Create the refinement node.
    
    Improves the answer if needed.
    """
    def refine(state: AgentState) -> AgentState:
        question = state["question"]
        context = state.get("context", "")
        current_answer = state.get("answer", "")
        
        # Refinement prompt
        prompt = refine_prompt.format(
            question=question,
            context=context,
            current_answer=current_answer,
        )
        
        refined_answer, _ = llm_fn("synthesis", 
            "You are an expert at improving answers. Make the answer more complete and accurate.",
            prompt,
        )
        
        return {
            **state,
            "answer": refined_answer,
            "refinement_count": state.get("refinement_count", 0) + 1,
            "needs_refinement": False,
            "steps_executed": ["refine"],
        }
    
    return refine


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------
def should_refine(state: AgentState) -> Literal["refine", "end"]:
    """Conditional edge: decide whether to refine or end."""
    if state.get("needs_refinement", False):
        if state.get("refinement_count", 0) < state.get("max_refinements", 1):
            return "refine"
    return "end"


def build_rag_agent(
    llm_fn,
    retrieve_fn,
    router_prompt: str,
    system_prompt: str,
    user_template: str,
    config: Optional[AgentConfig] = None,
) -> StateGraph:
    """
    Build the RAG agent graph.
    
    Graph structure:
        classify -> retrieve -> generate -> [refine] -> END
    
    Args:
        llm_fn: Function to call LLM (tier, system, user) -> (response, latency)
        retrieve_fn: Function to retrieve passages
        router_prompt: Prompt for classification
        system_prompt: System prompt for generation
        user_template: User prompt template for generation
        config: Agent configuration
        
    Returns:
        Compiled StateGraph
    """
    config = config or AgentConfig()
    
    # Refinement prompt
    refine_prompt = """The current answer may be incomplete or uncertain.

Question: {question}

Context:
{context}

Current Answer: {current_answer}

Please provide an improved, more complete answer based on the context."""
    
    # Create nodes
    classify_node = create_classify_node(llm_fn, router_prompt)
    retrieve_node = create_retrieve_node(retrieve_fn)
    generate_node = create_generate_node(llm_fn, system_prompt, user_template)
    refine_node = create_refine_node(llm_fn, refine_prompt)
    
    # Build graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("refine", refine_node)
    
    # Add edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    
    # Conditional edge after generate
    graph.add_conditional_edges(
        "generate",
        should_refine,
        {
            "refine": "refine",
            "end": END,
        },
    )
    
    # Refine always goes to end
    graph.add_edge("refine", END)
    
    return graph.compile()


# ---------------------------------------------------------------------------
# Agent Runner
# ---------------------------------------------------------------------------
class RAGAgent:
    """
    High-level RAG agent that wraps the LangGraph workflow.
    """
    
    def __init__(
        self,
        llm_fn,
        retrieve_fn,
        router_prompt: str,
        system_prompt: str,
        user_template: str,
        config: Optional[AgentConfig] = None,
    ):
        self.config = config or AgentConfig()
        self.graph = build_rag_agent(
            llm_fn=llm_fn,
            retrieve_fn=retrieve_fn,
            router_prompt=router_prompt,
            system_prompt=system_prompt,
            user_template=user_template,
            config=self.config,
        )
    
    def run(self, question: str) -> AgentResult:
        """
        Run the agent on a question.
        
        Args:
            question: The user's question
            
        Returns:
            AgentResult with answer, citations, and metadata
        """
        start = time.time()
        
        # Initial state
        initial_state: AgentState = {
            "question": question,
            "complexity": "simple",
            "retrieval_strategy": "semantic",
            "passages": [],
            "context": "",
            "answer": "",
            "citations": [],
            "needs_refinement": False,
            "refinement_count": 0,
            "max_refinements": self.config.max_refinements,
            "steps_executed": [],
            "latency_ms": 0.0,
            "model_used": "",
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        latency_ms = (time.time() - start) * 1000
        
        return AgentResult(
            answer=final_state.get("answer", ""),
            citations=final_state.get("citations", []),
            complexity=final_state.get("complexity", "simple"),
            retrieval_strategy=final_state.get("retrieval_strategy", "semantic"),
            steps_executed=final_state.get("steps_executed", []),
            latency_ms=latency_ms,
            model_used=final_state.get("model_used", ""),
            refinement_count=final_state.get("refinement_count", 0),
        )
