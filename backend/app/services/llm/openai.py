"""OpenAI LLM provider implementation."""

from typing import AsyncIterator
from openai import AsyncOpenAI

try:
    from app.config import settings
except ImportError:
    from backend.app.config import settings

from .base import LLMProvider


# RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context from documents.

Instructions:
- Answer the question using information from the context below
- Be helpful and provide relevant information even if the question isn't perfectly phrased
- If the context contains relevant information, use it to construct a helpful answer
- Only say you don't have information if the context truly contains nothing relevant to the question

Context:
{context}

Question: {question}

Answer:"""


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using GPT models."""

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the OpenAI provider.

        Args:
            model: The OpenAI model to use (default: gpt-4o).
        """
        super().__init__(model)
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    def _build_prompt(self, prompt: str, context: str) -> str:
        """Build the full prompt with context."""
        return RAG_PROMPT_TEMPLATE.format(context=context, question=prompt)

    async def generate(self, prompt: str, context: str) -> str:
        """
        Generate a response using OpenAI.

        Args:
            prompt: The user's question.
            context: Retrieved context from documents.

        Returns:
            The generated response text.
        """
        full_prompt = self._build_prompt(prompt, context)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )

        return response.choices[0].message.content or ""

    async def generate_stream(self, prompt: str, context: str) -> AsyncIterator[str]:
        """
        Stream a response using OpenAI.

        Args:
            prompt: The user's question.
            context: Retrieved context from documents.

        Yields:
            Chunks of the generated response.
        """
        full_prompt = self._build_prompt(prompt, context)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
