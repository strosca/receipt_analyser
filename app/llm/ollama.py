import logging

import httpx

from .base import LLMError, LLMResponse

log = logging.getLogger(__name__)


class OllamaClient:
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float, num_ctx: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx

    async def chat_json(self, system: str, user: str) -> LLMResponse:
        options: dict = {"temperature": 0}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        timeout = httpx.Timeout(self.timeout, connect=15)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama hat nicht innerhalb von {self.timeout:.0f} s geantwortet") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama nicht erreichbar ({self.base_url}): {exc}") from exc
        if r.status_code != 200:
            raise LLMError(f"Ollama-Fehler {r.status_code}: {r.text[:300]}")
        data = r.json()
        stats = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "seconds": round((data.get("total_duration") or 0) / 1e9, 1),
        }
        log.info("Ollama-Antwort: %s", stats)
        return LLMResponse(content=data.get("message", {}).get("content", ""), stats=stats)
