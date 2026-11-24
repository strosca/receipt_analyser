"""Austauschbare KI-Anbindung. Welcher Anbieter genutzt wird, steht in LLM_PROVIDER."""
import asyncio

from ..config import get_settings
from .base import LLMClient, LLMError, LLMResponse
from .ollama import OllamaClient
from .openai_compat import OpenAICompatClient

# Garantiert app-weit nur EINE laufende KI-Anfrage
llm_lock = asyncio.Lock()


def get_llm() -> LLMClient:
    s = get_settings()
    provider = s.llm_provider.lower().strip()
    if provider == "ollama":
        return OllamaClient(s.llm_base_url, s.llm_model, s.llm_timeout, s.llm_num_ctx, s.llm_num_thread, s.llm_think)
    if provider in ("openai", "openai-compatible", "openai_compat"):
        return OpenAICompatClient(s.llm_base_url, s.llm_model, s.llm_timeout, s.llm_api_key)
    raise LLMError(f"Unbekannter LLM_PROVIDER: {s.llm_provider!r} (erlaubt: ollama, openai)")


__all__ = ["LLMClient", "LLMError", "LLMResponse", "get_llm", "llm_lock"]
