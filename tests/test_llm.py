import asyncio
import json

from app.extraction import RESPONSE_SCHEMA, build_result, clean_text, parse_llm_response
from app.llm import LLMResponse
from app.llm.ollama import OllamaClient

from .test_extraction import REWE
from .test_lineparser import MUSTERBON


def test_short_keys_and_list_items():
    r = parse_llm_response('{"haendler":"REWE","datum":null,"summe":3.86,'
                           '"positionen":[{"n":"BANANEN","p":1.87},["LEERGUT",-0.25],{"n":"MILCH","p":"2,24"}]}')
    assert r.gesamtsumme == 3.86
    assert [(i.name, i.preis) for i in r.positionen] == [("BANANEN", 1.87), ("LEERGUT", -0.25), ("MILCH", 2.24)]


def test_clean_text_stops_after_total():
    text, _ = clean_text(MUSTERBON, 10_000)
    assert "SUMME" in text and "Geg. EC-Karte" in text
    assert "Steuer" not in text and "Terminal" not in text and "girocard" not in text
    assert len(text) < len(MUSTERBON) * 0.7


def test_payment_and_branch_come_from_rules_when_llm_omits_them():
    parsed = parse_llm_response('{"haendler":"MUSTERMANN","summe":24.66,"positionen":[{"n":"X","p":24.66}]}')
    res = build_result(parsed, MUSTERBON, 0.05)
    assert res.payment_method == "Karte"
    assert res.branch == "Musterstrasse 12, 12345 Musterstadt"
    assert res.merchant == "FRISCHMARKT MUSTERMANN"
    assert res.purchase_date == "2026-11-24"


def test_ollama_payload_options():
    c = OllamaClient("http://ollama:11434/", "qwen-bons", 60, num_ctx=8192, num_thread=3, think=False)
    p = c.build_payload("sys", "user", RESPONSE_SCHEMA)
    assert p["format"] == RESPONSE_SCHEMA and p["think"] is False
    assert p["options"] == {"temperature": 0, "num_ctx": 8192, "num_thread": 3}
    plain = OllamaClient("http://x", "m", 60).build_payload("s", "u", None)
    assert plain["format"] == "json" and "think" not in plain and plain["options"] == {"temperature": 0}


class _Resp:
    def __init__(self, status, data=None, text=""):
        self.status_code, self._data, self.text = status, data, text

    def json(self):
        return self._data


def test_ollama_falls_back_to_plain_json_when_schema_rejected():
    c = OllamaClient("http://x", "m", 60)
    sent = []

    async def fake_post(payload):
        sent.append(payload["format"])
        if isinstance(payload["format"], dict):
            return _Resp(400, text="invalid format")
        return _Resp(200, {"message": {"content": '{"ok":true}'}, "eval_count": 6, "eval_duration": 3.5e9})

    c._post = fake_post
    resp = asyncio.run(c.chat_json("s", "u", RESPONSE_SCHEMA))
    assert isinstance(resp, LLMResponse) and json.loads(resp.content) == {"ok": True}
    assert sent == [RESPONSE_SCHEMA, "json"]
    assert resp.stats["tokens_per_s"] == 1.71
