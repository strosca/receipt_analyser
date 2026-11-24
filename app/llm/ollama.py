import logging

import httpx

from .base import LLMError, LLMResponse

log = logging.getLogger(__name__)


class OllamaClient:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        num_ctx: int | None = None,
        num_thread: int | None = None,
        think: bool | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_thread = num_thread
        self.think = think

    def build_payload(self, system: str, user: str, schema: dict | None) -> dict:
        options: dict = {"temperature": 0}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if self.num_thread:
            options["num_thread"] = self.num_thread
        payload = {
            "model": self.model,
            "stream": False,
            # Mit Schema erzwingt Ollama genau diese JSON-Struktur, sonst nur "irgendein JSON"
            "format": schema or "json",
            "options": options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload

    async def _post(self, payload: dict) -> httpx.Response:
        timeout = httpx.Timeout(self.timeout, connect=15)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama hat nicht innerhalb von {self.timeout:.0f} s geantwortet") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama nicht erreichbar ({self.base_url}): {exc}") from exc

    async def chat_json(self, system: str, user: str, schema: dict | None = None) -> LLMResponse:
        payload = self.build_payload(system, user, schema)
        r = await self._post(payload)
        if r.status_code == 400 and schema is not None:
            # Ältere Ollama-Versionen kennen kein Schema -> einfaches JSON-Format
            log.warning("Ollama lehnt JSON-Schema ab (%s), nutze format=json", r.text[:200])
            payload["format"] = "json"
            r = await self._post(payload)
        if r.status_code != 200:
            raise LLMError(f"Ollama-Fehler {r.status_code}: {r.text[:300]}")
        data = r.json()
        stats = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "seconds": round((data.get("total_duration") or 0) / 1e9, 1),
            "load_seconds": round((data.get("load_duration") or 0) / 1e9, 1),
        }
        if data.get("eval_count") and data.get("eval_duration"):
            stats["tokens_per_s"] = round(data["eval_count"] / (data["eval_duration"] / 1e9), 2)
        log.info("Ollama-Antwort: %s", stats)
        return LLMResponse(content=data.get("message", {}).get("content", ""), stats=stats)
