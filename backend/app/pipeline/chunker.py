"""
Document chunking strategies for the RAG pipeline.

Supports:
- Recursive text splitting (paragraph → sentence → word)
- Configurable chunk size and overlap
- Metadata preservation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import hashlib
import re

from ..config import get_config, ChunkingConfig


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    
    # Source tracking
    source_id: Optional[str] = None
    chunk_index: int = 0
    
    # Position in original document
    start_char: int = 0
    end_char: int = 0
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on content hash."""
        content = f"{self.source_id}:{self.chunk_index}:{self.text[:50]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    @property
    def char_count(self) -> int:
        """Number of characters in the chunk."""
        return len(self.text)
    
    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.text.split())


class Chunker(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, source_id: Optional[str] = None, metadata: Optional[dict] = None) -> List[Chunk]:
        """Split text into chunks."""
        pass
    
    @abstractmethod
    def chunk_documents(self, documents: List[dict]) -> List[Chunk]:
        """Chunk multiple documents."""
        pass


class RecursiveChunker(Chunker):
    """
    Recursive text splitter that tries to split on natural boundaries.
    
    Attempts to split on paragraphs first, then sentences, then words,
    ensuring chunks stay within the configured size limits.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize with optional custom config."""
        self.config = config or get_config().chunking
    
    def chunk(
        self,
        text: str,
        source_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        Split text into chunks using recursive splitting.
        
        Args:
            text: The text to chunk
            source_id: Optional identifier for the source document
            metadata: Optional metadata to attach to all chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        # Normalize whitespace
        text = self._normalize_text(text)
        
        # Recursively split
        raw_chunks = self._recursive_split(text, self.config.separators)
        
        # Merge small chunks and create Chunk objects
        chunks = self._merge_and_create_chunks(
            raw_chunks,
            source_id=source_id,
            metadata=metadata or {},
        )
        
        return chunks
    
    def chunk_documents(self, documents: List[dict]) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'
            
        Returns:
            List of all Chunk objects from all documents
        """
        all_chunks = []
        
        for doc in documents:
            doc_id = doc.get("id", "")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            chunks = self.chunk(text, source_id=doc_id, metadata=metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using separators in order of priority.
        
        Args:
            text: Text to split
            separators: List of separators to try, in order
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.config.chunk_size:
            return [text] if text.strip() else []
        
        if not separators:
            # No more separators, force split by character
            return self._force_split(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split on current separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means split by character
            splits = list(text)
        
        # Process splits
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Add separator back (except for empty separator)
            piece = split + separator if separator else split
            
            if len(current_chunk) + len(piece) <= self.config.chunk_size:
                current_chunk += piece
            else:
                # Current chunk is full
                if current_chunk:
                    # Recursively split if still too large
                    if len(current_chunk) > self.config.chunk_size:
                        chunks.extend(self._recursive_split(current_chunk, remaining_separators))
                    else:
                        chunks.append(current_chunk.rstrip(separator))
                
                # Start new chunk
                if len(piece) > self.config.chunk_size:
                    # Piece itself is too large, recursively split
                    chunks.extend(self._recursive_split(piece, remaining_separators))
                    current_chunk = ""
                else:
                    current_chunk = piece
        
        # Don't forget the last chunk
        if current_chunk:
            if len(current_chunk) > self.config.chunk_size:
                chunks.extend(self._recursive_split(current_chunk, remaining_separators))
            else:
                chunks.append(current_chunk.rstrip(separator))
        
        return [c for c in chunks if c.strip()]
    
    def _force_split(self, text: str) -> List[str]:
        """Force split text into chunk_size pieces."""
        chunks = []
        for i in range(0, len(text), self.config.chunk_size):
            chunk = text[i:i + self.config.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def _merge_and_create_chunks(
        self,
        raw_chunks: List[str],
        source_id: Optional[str],
        metadata: dict,
    ) -> List[Chunk]:
        """
        Merge small chunks and create Chunk objects with overlap.
        
        Args:
            raw_chunks: List of raw text chunks
            source_id: Source document ID
            metadata: Metadata to attach
            
        Returns:
            List of Chunk objects
        """
        if not raw_chunks:
            return []
        
        chunks = []
        current_pos = 0
        
        for i, text in enumerate(raw_chunks):
            # Skip chunks that are too small (unless it's the only chunk)
            if len(text) < self.config.min_chunk_size and len(raw_chunks) > 1:
                # Try to merge with next chunk
                if i + 1 < len(raw_chunks):
                    raw_chunks[i + 1] = text + " " + raw_chunks[i + 1]
                    continue
            
            # Create chunk with overlap from previous
            chunk_text = text
            if chunks and self.config.chunk_overlap > 0:
                # Add overlap from end of previous chunk
                prev_text = chunks[-1].text
                overlap = prev_text[-self.config.chunk_overlap:]
                # Only add if it doesn't make chunk too large
                if len(overlap) + len(chunk_text) <= self.config.chunk_size * 1.2:
                    chunk_text = overlap + " " + chunk_text
            
            chunk = Chunk(
                id="",  # Will be auto-generated
                text=chunk_text.strip(),
                metadata={**metadata, "chunk_index": i},
                source_id=source_id,
                chunk_index=i,
                start_char=current_pos,
                end_char=current_pos + len(text),
            )
            chunks.append(chunk)
            current_pos += len(text) + 1  # +1 for separator
        
        return chunks


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    source_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    config: Optional[ChunkingConfig] = None,
) -> List[Chunk]:
    """
    Convenience function to chunk text with default settings.
    
    Args:
        text: Text to chunk
        source_id: Optional source document ID
        metadata: Optional metadata
        config: Optional custom chunking config
        
    Returns:
        List of Chunk objects
    """
    chunker = RecursiveChunker(config)
    return chunker.chunk(text, source_id, metadata)
