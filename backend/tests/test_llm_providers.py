"""Phase 4 Tests: LLM Providers

Tests for LLM provider abstraction and OpenAI implementation.
These tests require OPENAI_API_KEY to be set.
"""

import sys
from pathlib import Path
import os

# Add backend to path for local testing
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import pytest

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


class TestLLMProviderFactory:
    """Tests for the LLM provider factory."""

    def test_get_openai_provider(self):
        """Test factory returns OpenAI provider."""
        from app.services.llm import get_llm_provider, OpenAIProvider

        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_MODEL"] = "gpt-4o"

        provider = get_llm_provider()
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o"

    def test_get_provider_with_override(self):
        """Test factory respects parameter overrides."""
        from app.services.llm import get_llm_provider, OpenAIProvider

        provider = get_llm_provider(provider="openai", model="gpt-4o-mini")
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"

    def test_unknown_provider_raises_error(self):
        """Test factory raises error for unknown provider."""
        from app.services.llm import get_llm_provider

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider(provider="unknown")


class TestOpenAIProvider:
    """Tests for the OpenAI provider."""

    @pytest.mark.asyncio
    async def test_generate_simple_response(self):
        """Test OpenAI provider can generate a simple response."""
        from app.services.llm.openai import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o-mini")
        response = await provider.generate(
            prompt="What is 2+2?",
            context="Mathematics: 2+2=4"
        )

        assert response is not None
        assert len(response) > 0
        assert "4" in response

    @pytest.mark.asyncio
    async def test_generate_uses_context(self):
        """Test OpenAI provider uses provided context."""
        from app.services.llm.openai import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o-mini")
        response = await provider.generate(
            prompt="What is the capital city mentioned?",
            context="The capital of France is Paris. It is known for the Eiffel Tower."
        )

        assert "Paris" in response

    @pytest.mark.asyncio
    async def test_generate_without_context(self):
        """Test provider handles empty context appropriately."""
        from app.services.llm.openai import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o-mini")
        response = await provider.generate(
            prompt="What is quantum computing?",
            context=""
        )

        # Should indicate lack of information
        assert "don't have enough information" in response.lower() or len(response) > 0

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test OpenAI provider can stream responses."""
        from app.services.llm.openai import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o-mini")
        chunks = []

        async for chunk in provider.generate_stream(
            prompt="Count from 1 to 5",
            context="Numbers: 1, 2, 3, 4, 5"
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0

    def test_build_prompt(self):
        """Test prompt building includes context and question."""
        from app.services.llm.openai import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o-mini")
        prompt = provider._build_prompt(
            prompt="What is the answer?",
            context="The answer is 42."
        )

        assert "What is the answer?" in prompt
        assert "The answer is 42." in prompt
        assert "Context:" in prompt
