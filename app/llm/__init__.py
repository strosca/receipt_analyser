"""KI-Anbindung: ausschließlich das lokale Ollama im Heimnetz.

Bewusst gibt es keine Anbindung an öffentliche KI-Dienste – Bons verlassen nie das eigene Netz.
"""
import asyncio

from ..config import get_settings
from .base import LLMClient, LLMError, LLMResponse
from .ollama import OllamaClient

# Garantiert app-weit nur EINE laufende KI-Anfrage
llm_lock = asyncio.Lock()


def get_llm() -> LLMClient:
    s = get_settings()
    return OllamaClient(s.llm_base_url, s.llm_model, s.llm_timeout, s.llm_num_ctx, s.llm_num_thread, s.llm_think)


__all__ = ["LLMClient", "LLMError", "LLMResponse", "OllamaClient", "get_llm", "llm_lock"]
