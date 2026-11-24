import logging

import httpx

from .base import LLMError, LLMResponse

log = logging.getLogger(__name__)


class OpenAICompatClient:
    """Jede API, die /v1/chat/completions wie OpenAI anbietet (OpenAI, LM Studio, vLLM, …)."""

    name = "openai"

    def __init__(self, base_url: str, model: str, timeout: float, api_key: str | None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    async def chat_json(self, system: str, user: str) -> LLMResponse:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=15)) as client:
                r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMError(f"KI-API hat nicht innerhalb von {self.timeout:.0f} s geantwortet") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"KI-API nicht erreichbar ({self.base_url}): {exc}") from exc
        if r.status_code != 200:
            raise LLMError(f"KI-API-Fehler {r.status_code}: {r.text[:300]}")
        data = r.json()
        usage = data.get("usage") or {}
        stats = {"prompt_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError("Unerwartetes Antwortformat der KI-API") from exc
        return LLMResponse(content=content or "", stats=stats)
