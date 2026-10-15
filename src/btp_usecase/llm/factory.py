from __future__ import annotations

from ..core.config import Settings
from .base import LLMProvider
from .ollama_provider import OllamaProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider.lower() == "ollama":
        return OllamaProvider(base_url=settings.ollama_base_url, model=settings.llm_model)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
