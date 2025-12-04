"""
Tests for the tokenizer module.
"""

import pytest
from app.pipeline.tokenizer import (
    TokenCounter,
    TokenUsage,
    get_token_counter,
    count_tokens,
    count_usage,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""
    
    def test_token_usage_creation(self):
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
    
    def test_cost_estimate(self):
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=1000,
            total_tokens=2000,
        )
        
        # GPT-4 pricing: $0.01/1K input, $0.03/1K output
        expected = (1000 / 1000) * 0.01 + (1000 / 1000) * 0.03
        assert usage.cost_estimate_usd == expected


class TestTokenCounter:
    """Tests for TokenCounter."""
    
    def test_count_empty_string(self):
        counter = TokenCounter()
        assert counter.count("") == 0
    
    def test_count_simple_text(self):
        counter = TokenCounter()
        tokens = counter.count("Hello, world!")
        
        # Should be a small number of tokens
        assert tokens > 0
        assert tokens < 10
    
    def test_count_longer_text(self):
        counter = TokenCounter()
        short = counter.count("Hello")
        long = counter.count("Hello, this is a much longer sentence with more words.")
        
        assert long > short
    
    def test_count_messages(self):
        counter = TokenCounter()
        
        system = "You are a helpful assistant."
        user = "What is the capital of France?"
        
        tokens = counter.count_messages(system, user)
        
        # Should include overhead for message formatting
        base_tokens = counter.count(system) + counter.count(user)
        assert tokens > base_tokens
    
    def test_count_usage(self):
        counter = TokenCounter()
        
        usage = counter.count_usage(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2+2?",
            response="The answer is 4.",
        )
        
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    
    def test_truncate_to_limit(self):
        counter = TokenCounter()
        
        long_text = "This is a very long text. " * 100
        truncated = counter.truncate_to_limit(long_text, max_tokens=20)
        
        # Truncated should be shorter
        assert len(truncated) < len(long_text)
        assert truncated.endswith("...")
    
    def test_truncate_short_text_unchanged(self):
        counter = TokenCounter()
        
        short_text = "Hello"
        result = counter.truncate_to_limit(short_text, max_tokens=100)
        
        assert result == short_text


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_token_counter_singleton(self):
        counter1 = get_token_counter()
        counter2 = get_token_counter()
        
        assert counter1 is counter2
    
    def test_count_tokens(self):
        tokens = count_tokens("Hello, world!")
        
        assert tokens > 0
        assert tokens < 10
    
    def test_count_usage_function(self):
        usage = count_usage(
            system_prompt="System",
            user_prompt="User",
            response="Response",
        )
        
        assert isinstance(usage, TokenUsage)
        assert usage.total_tokens > 0
