"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

try:
    from app.config import settings
except ImportError:
    from backend.app.config import settings


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str):
        """
        Initialize the LLM provider.

        Args:
            model: The model name to use.
        """
        self.model = model

    @abstractmethod
    async def generate(self, prompt: str, context: str) -> str:
        """
        Generate a response given prompt and context.

        Args:
            prompt: The user's question or prompt.
            context: Retrieved context from the vector store.

        Returns:
            The generated response text.
        """
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, context: str) -> AsyncIterator[str]:
        """
        Stream a response given prompt and context.

        Args:
            prompt: The user's question or prompt.
            context: Retrieved context from the vector store.

        Yields:
            Chunks of the generated response.
        """
        pass


def get_llm_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> LLMProvider:
    """
    Factory function to get configured LLM provider.

    Args:
        provider: Override for LLM_PROVIDER setting.
        model: Override for LLM_MODEL setting.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If the provider is unknown.
    """
    provider = provider or settings.LLM_PROVIDER
    model = model or settings.LLM_MODEL

    if provider == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(model=model)
    elif provider == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(model=model)
    elif provider == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
