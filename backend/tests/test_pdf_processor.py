"""Phase 2 Tests: PDF Processing

Tests for PDF text extraction and chunking functionality.
"""

import sys
from pathlib import Path

# Add backend to path for local testing
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import pytest
from app.services.pdf_processor import (
    chunk_text,
    count_tokens,
    _split_into_sentences,
)


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_chunk_text_empty_string(self):
        """Test chunking with empty string returns empty list."""
        chunks = chunk_text("")
        assert chunks == []

    def test_chunk_text_whitespace_only(self):
        """Test chunking with whitespace only returns empty list."""
        chunks = chunk_text("   \n\n   ")
        assert chunks == []

    def test_chunk_text_single_paragraph(self):
        """Test chunking with a single short paragraph."""
        text = "This is a simple test paragraph."
        chunks = chunk_text(text, chunk_size=500, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_produces_multiple_chunks(self):
        """Test that long text produces multiple chunks."""
        # Create text that's definitely longer than one chunk
        text = "Lorem ipsum dolor sit amet. " * 200
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        assert len(chunks) > 1

    def test_chunk_text_respects_max_size(self):
        """Test that chunks respect the maximum size limit."""
        text = "Word " * 1000
        chunk_size = 100
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=0)

        for chunk in chunks:
            token_count = count_tokens(chunk)
            # Allow some overflow due to sentence/paragraph boundaries
            assert token_count <= chunk_size * 1.5

    def test_chunk_text_overlap(self):
        """Test that chunks have proper overlap."""
        # Create text with distinct numbered words
        words = [f"word{i}" for i in range(500)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=50, overlap=10)

        # Check that there's overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk_words = set(chunks[i].split()[-20:])
            next_words = set(chunks[i + 1].split()[:20])
            # Should have some overlap
            overlap = chunk_words & next_words
            assert len(overlap) > 0, f"No overlap between chunk {i} and {i+1}"

    def test_chunk_text_preserves_paragraphs(self):
        """Test that chunking preserves paragraph boundaries when possible."""
        text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph."
        chunks = chunk_text(text, chunk_size=500, overlap=0)

        # With large chunk size, should keep paragraphs together
        assert len(chunks) == 1
        assert "First paragraph" in chunks[0]
        assert "Second paragraph" in chunks[0]

    def test_chunk_text_handles_long_paragraphs(self):
        """Test that long paragraphs are split properly."""
        # Single very long paragraph
        text = "This is a sentence. " * 200
        chunks = chunk_text(text, chunk_size=50, overlap=0)

        # Should produce multiple chunks
        assert len(chunks) > 1

        # Each chunk should have content
        for chunk in chunks:
            assert len(chunk.strip()) > 0


class TestCountTokens:
    """Tests for the count_tokens function."""

    def test_count_tokens_empty(self):
        """Test token count for empty string."""
        assert count_tokens("") == 0

    def test_count_tokens_simple(self):
        """Test token count for simple text."""
        # "Hello world" is typically 2 tokens
        count = count_tokens("Hello world")
        assert count > 0
        assert count < 10

    def test_count_tokens_longer_text(self):
        """Test token count for longer text."""
        text = "This is a longer piece of text that should have more tokens."
        count = count_tokens(text)
        assert count > 5


class TestSplitIntoSentences:
    """Tests for sentence splitting."""

    def test_split_simple_sentences(self):
        """Test splitting simple sentences."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3

    def test_split_with_question_mark(self):
        """Test splitting with question marks."""
        text = "Is this a question? Yes it is. Another statement."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3

    def test_split_with_exclamation(self):
        """Test splitting with exclamation marks."""
        text = "Wow! That's amazing. Indeed it is!"
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3

    def test_split_empty_string(self):
        """Test splitting empty string."""
        sentences = _split_into_sentences("")
        assert sentences == []


class TestIntegration:
    """Integration tests for PDF processing pipeline."""

    def test_chunk_and_count_consistency(self):
        """Test that chunked text token counts are reasonable."""
        text = "Sample text for testing. " * 100
        chunk_size = 50

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=0)

        # Total tokens in chunks should be close to original
        original_tokens = count_tokens(text)
        chunked_tokens = sum(count_tokens(c) for c in chunks)

        # Chunked tokens might be slightly more due to overlap, but shouldn't be drastically different
        assert chunked_tokens >= original_tokens * 0.9
