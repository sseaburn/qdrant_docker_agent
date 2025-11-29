"""Ollama LLM provider implementation (placeholder)."""

from typing import AsyncIterator

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local models."""

    def __init__(self, model: str = "llama3.2"):
        """
        Initialize the Ollama provider.

        Args:
            model: The Ollama model to use.
        """
        super().__init__(model)
        # TODO: Initialize Ollama client when needed
        raise NotImplementedError("Ollama provider not yet implemented")

    async def generate(self, prompt: str, context: str) -> str:
        """Generate a response using Ollama."""
        raise NotImplementedError("Ollama provider not yet implemented")

    async def generate_stream(self, prompt: str, context: str) -> AsyncIterator[str]:
        """Stream a response using Ollama."""
        raise NotImplementedError("Ollama provider not yet implemented")
        yield  # Makes this a generator
