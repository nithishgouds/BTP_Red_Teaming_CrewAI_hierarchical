from __future__ import annotations

import httpx

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, *, prompt: str, context: str) -> str:
        payload = {
            "model": self.model,
            "prompt": f"Use the following document context to answer the question.\n\nQuestion: {prompt}\n\nContext:\n{context}",
            "stream": False,
        }

        try:
            response = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=20.0)
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            if text:
                return text
            return "No response returned from Ollama."
        except httpx.HTTPError:
            return (
                f"Ollama is not available at {self.base_url}. "
                f"Using a local fallback response for the prompt: '{prompt}'."
            )
