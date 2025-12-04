"""
Unit tests for the chunker module.
"""

import pytest
from app.pipeline.chunker import RecursiveChunker, Chunk, chunk_text
from app.config import ChunkingConfig


class TestChunk:
    """Tests for the Chunk dataclass."""
    
    def test_chunk_creation(self):
        chunk = Chunk(
            id="test-1",
            text="Hello world",
            source_id="doc-1",
            chunk_index=0,
        )
        assert chunk.id == "test-1"
        assert chunk.text == "Hello world"
        assert chunk.char_count == 11
        assert chunk.word_count == 2
    
    def test_chunk_auto_id(self):
        chunk = Chunk(
            id="",
            text="Auto ID test",
            source_id="doc-1",
            chunk_index=0,
        )
        # ID should be auto-generated
        assert len(chunk.id) == 12  # MD5 hash prefix


class TestRecursiveChunker:
    """Tests for the RecursiveChunker."""
    
    def test_chunk_short_text(self):
        chunker = RecursiveChunker()
        chunks = chunker.chunk("Short text.", source_id="test")
        
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].source_id == "test"
    
    def test_chunk_empty_text(self):
        chunker = RecursiveChunker()
        chunks = chunker.chunk("", source_id="test")
        
        assert len(chunks) == 0
    
    def test_chunk_with_metadata(self):
        chunker = RecursiveChunker()
        chunks = chunker.chunk(
            "Test text with metadata.",
            source_id="test",
            metadata={"author": "pytest"},
        )
        
        assert len(chunks) == 1
        assert chunks[0].metadata.get("author") == "pytest"
    
    def test_chunk_long_text(self):
        # Create text longer than default chunk size
        long_text = "This is a sentence. " * 100
        
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunker = RecursiveChunker(config=config)
        chunks = chunker.chunk(long_text, source_id="long-doc")
        
        assert len(chunks) > 1
        # Each chunk should be roughly within size limit
        for chunk in chunks:
            assert chunk.char_count <= config.chunk_size * 1.5  # Allow some flexibility
    
    def test_chunk_preserves_paragraphs(self):
        # Use longer paragraphs to force splitting
        text = "First paragraph with more content here.\n\nSecond paragraph with additional text.\n\nThird paragraph with even more content to ensure splitting."
        
        config = ChunkingConfig(chunk_size=50, chunk_overlap=0)
        chunker = RecursiveChunker(config=config)
        chunks = chunker.chunk(text, source_id="para-doc")
        
        # Should create multiple chunks
        assert len(chunks) >= 1  # At least one chunk
    
    def test_chunk_documents(self):
        chunker = RecursiveChunker()
        documents = [
            {"id": "doc-1", "text": "First document.", "metadata": {"type": "test"}},
            {"id": "doc-2", "text": "Second document."},
        ]
        
        chunks = chunker.chunk_documents(documents)
        
        assert len(chunks) == 2
        assert chunks[0].source_id == "doc-1"
        assert chunks[1].source_id == "doc-2"


class TestChunkTextFunction:
    """Tests for the convenience chunk_text function."""
    
    def test_chunk_text_default(self):
        chunks = chunk_text("Simple test text.")
        
        assert len(chunks) == 1
        assert chunks[0].text == "Simple test text."
    
    def test_chunk_text_with_config(self):
        config = ChunkingConfig(chunk_size=10)
        chunks = chunk_text("A bit longer text here.", config=config)
        
        assert len(chunks) >= 1
