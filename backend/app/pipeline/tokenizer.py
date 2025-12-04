"""
Token counting utilities using tiktoken.

Provides accurate token counting for:
- Input prompts (system + user)
- Output responses
- Cost estimation

Uses cl100k_base encoding (GPT-4/ChatGPT compatible) as a reasonable
approximation for local models like Qwen, Llama, etc.
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import tiktoken


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    @property
    def cost_estimate_usd(self) -> float:
        """
        Rough cost estimate based on GPT-4 pricing.
        
        For local models this is $0, but useful for comparison
        if migrating to cloud APIs.
        
        GPT-4 Turbo: $0.01/1K input, $0.03/1K output
        """
        input_cost = (self.input_tokens / 1000) * 0.01
        output_cost = (self.output_tokens / 1000) * 0.03
        return round(input_cost + output_cost, 6)


class TokenCounter:
    """
    Token counter using tiktoken.
    
    Uses cl100k_base encoding which is compatible with GPT-4 and
    provides a reasonable approximation for other models.
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize token counter.
        
        Args:
            encoding_name: Tiktoken encoding name (default: cl100k_base)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
    
    def count(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4",
    ) -> int:
        """
        Count tokens for a chat message format.
        
        Accounts for message formatting overhead.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            model: Model name (for format-specific overhead)
            
        Returns:
            Total input tokens including overhead
        """
        # Base token count
        tokens = self.count(system_prompt) + self.count(user_prompt)
        
        # Add overhead for message formatting
        # Each message has ~4 tokens overhead (role, content markers)
        # Plus ~3 tokens for the overall structure
        overhead = 4 * 2 + 3  # 2 messages + structure
        
        return tokens + overhead
    
    def count_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        response: str,
    ) -> TokenUsage:
        """
        Count full token usage for a request/response pair.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            response: Model response
            
        Returns:
            TokenUsage with input, output, and total counts
        """
        input_tokens = self.count_messages(system_prompt, user_prompt)
        output_tokens = self.count(response)
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    
    def truncate_to_limit(
        self,
        text: str,
        max_tokens: int,
        suffix: str = "...",
    ) -> str:
        """
        Truncate text to fit within a token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated text
        """
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Reserve space for suffix
        suffix_tokens = self.count(suffix)
        truncate_to = max_tokens - suffix_tokens
        
        if truncate_to <= 0:
            return suffix
        
        truncated_tokens = tokens[:truncate_to]
        return self.encoding.decode(truncated_tokens) + suffix


# ---------------------------------------------------------------------------
# Global instance (singleton)
# ---------------------------------------------------------------------------
_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get or create the global token counter."""
    global _counter
    if _counter is None:
        _counter = TokenCounter()
    return _counter


def count_tokens(text: str) -> int:
    """Convenience function to count tokens."""
    return get_token_counter().count(text)


def count_usage(
    system_prompt: str,
    user_prompt: str,
    response: str,
) -> TokenUsage:
    """Convenience function to count full usage."""
    return get_token_counter().count_usage(system_prompt, user_prompt, response)
