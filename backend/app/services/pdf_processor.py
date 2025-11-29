"""PDF processing service for text extraction and chunking."""

import io
import re
import unicodedata
from typing import List
from pypdf import PdfReader
import tiktoken


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text by fixing common extraction issues.

    Args:
        text: Raw extracted text from PDF.

    Returns:
        Cleaned text.
    """
    if not text:
        return text

    # Remove control characters (except newlines, tabs)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Cc' or char in '\n\r\t')

    # Fix common ligature extraction issues
    ligature_fixes = {
        '/f_i': 'fi',
        '/f_l': 'fl',
        '/f_': 'f',
        '/T_': 'Th',
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
    }
    for bad, good in ligature_fixes.items():
        text = text.replace(bad, good)

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Fix hyphenation at line breaks (word-\n continuation)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double

    return text.strip()


def extract_text(file_content: bytes) -> str:
    """
    Extract text from PDF file bytes.

    Args:
        file_content: Raw bytes of the PDF file.

    Returns:
        Extracted text as a string.
    """
    pdf = PdfReader(io.BytesIO(file_content))
    text_parts = []

    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    raw_text = "\n".join(text_parts).strip()
    return clean_text(raw_text)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: The text to count tokens for.
        encoding_name: The tiktoken encoding to use.

    Returns:
        Number of tokens.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Split text into overlapping chunks based on token count.

    Attempts to preserve paragraph boundaries where possible.

    Args:
        text: The text to split into chunks.
        chunk_size: Target size of each chunk in tokens.
        overlap: Number of tokens to overlap between chunks.
        encoding_name: The tiktoken encoding to use.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    encoding = tiktoken.get_encoding(encoding_name)

    # Split by paragraphs first (double newline)
    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk_tokens = []
    current_chunk_text = []

    for paragraph in paragraphs:
        paragraph_tokens = encoding.encode(paragraph)

        # If adding this paragraph exceeds chunk size
        if len(current_chunk_tokens) + len(paragraph_tokens) > chunk_size:
            # Save current chunk if it has content
            if current_chunk_text:
                chunk_text_str = "\n\n".join(current_chunk_text)
                chunks.append(chunk_text_str)

                # Calculate overlap
                if overlap > 0 and len(current_chunk_tokens) > overlap:
                    # Keep last 'overlap' tokens worth of text
                    overlap_text = encoding.decode(current_chunk_tokens[-overlap:])
                    current_chunk_tokens = encoding.encode(overlap_text)
                    current_chunk_text = [overlap_text]
                else:
                    current_chunk_tokens = []
                    current_chunk_text = []

        # If single paragraph is larger than chunk size, split it
        if len(paragraph_tokens) > chunk_size:
            # Split long paragraph by sentences or fixed token windows
            sentences = _split_into_sentences(paragraph)

            # If there's only one "sentence" (no natural boundaries), split by tokens
            if len(sentences) == 1 and len(paragraph_tokens) > chunk_size:
                token_chunks = _split_by_tokens(paragraph, chunk_size, encoding_name)
                for token_chunk in token_chunks:
                    chunk_tokens = encoding.encode(token_chunk)

                    if len(current_chunk_tokens) + len(chunk_tokens) > chunk_size:
                        if current_chunk_text:
                            chunk_text_str = " ".join(current_chunk_text)
                            chunks.append(chunk_text_str)

                            if overlap > 0 and len(current_chunk_tokens) > overlap:
                                overlap_text = encoding.decode(current_chunk_tokens[-overlap:])
                                current_chunk_tokens = encoding.encode(overlap_text)
                                current_chunk_text = [overlap_text]
                            else:
                                current_chunk_tokens = []
                                current_chunk_text = []

                    current_chunk_tokens.extend(chunk_tokens)
                    current_chunk_text.append(token_chunk)
            else:
                for sentence in sentences:
                    sentence_tokens = encoding.encode(sentence)

                    if len(current_chunk_tokens) + len(sentence_tokens) > chunk_size:
                        if current_chunk_text:
                            chunk_text_str = " ".join(current_chunk_text) if len(current_chunk_text) > 1 else current_chunk_text[0]
                            chunks.append(chunk_text_str)

                            if overlap > 0 and len(current_chunk_tokens) > overlap:
                                overlap_text = encoding.decode(current_chunk_tokens[-overlap:])
                                current_chunk_tokens = encoding.encode(overlap_text)
                                current_chunk_text = [overlap_text]
                            else:
                                current_chunk_tokens = []
                                current_chunk_text = []

                    current_chunk_tokens.extend(sentence_tokens)
                    current_chunk_text.append(sentence)
        else:
            current_chunk_tokens.extend(paragraph_tokens)
            current_chunk_text.append(paragraph)

    # Don't forget the last chunk
    if current_chunk_text:
        chunk_text_str = "\n\n".join(current_chunk_text)
        chunks.append(chunk_text_str)

    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Text to split.

    Returns:
        List of sentences.
    """
    # Simple sentence splitting by common delimiters
    import re

    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = [s.strip() for s in sentences if s.strip()]

    # If no sentences were found (no punctuation), return the original text as one item
    if not result and text.strip():
        return [text.strip()]

    return result


def _split_by_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> List[str]:
    """
    Split text into chunks by token count when no natural boundaries exist.

    Args:
        text: Text to split.
        max_tokens: Maximum tokens per chunk.
        encoding_name: The tiktoken encoding to use.

    Returns:
        List of text chunks.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))

    return chunks


def process_pdf(file_content: bytes, chunk_size: int = 500, overlap: int = 100) -> dict:
    """
    Process a PDF file: extract text and create chunks.

    Args:
        file_content: Raw bytes of the PDF file.
        chunk_size: Target size of each chunk in tokens.
        overlap: Number of tokens to overlap between chunks.

    Returns:
        Dictionary with extracted text and chunks.
    """
    text = extract_text(file_content)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    return {
        "text": text,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "total_tokens": count_tokens(text)
    }
