"""Anthropic LLM provider implementation (placeholder)."""

from typing import AsyncIterator

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider using Claude models."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the Anthropic provider.

        Args:
            model: The Anthropic model to use.
        """
        super().__init__(model)
        # TODO: Initialize Anthropic client when needed
        raise NotImplementedError("Anthropic provider not yet implemented")

    async def generate(self, prompt: str, context: str) -> str:
        """Generate a response using Anthropic."""
        raise NotImplementedError("Anthropic provider not yet implemented")

    async def generate_stream(self, prompt: str, context: str) -> AsyncIterator[str]:
        """Stream a response using Anthropic."""
        raise NotImplementedError("Anthropic provider not yet implemented")
        yield  # Makes this a generator
