from dataclasses import dataclass, field
from typing import Protocol


class LLMError(Exception):
    pass


@dataclass
class LLMResponse:
    content: str
    stats: dict = field(default_factory=dict)  # z. B. Token-Zahlen, Dauer


class LLMClient(Protocol):
    name: str

    async def chat_json(self, system: str, user: str, schema: dict | None = None) -> LLMResponse:
        """Schickt System- und Benutzer-Nachricht und erwartet eine JSON-Antwort (als Text).

        schema: optionales JSON-Schema, an das sich die Antwort halten soll.
        """
        ...
