from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, *, prompt: str, context: str) -> str:
        """Generate a response using the current model/provider."""
