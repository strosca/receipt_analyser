import asyncio
import json

import pytest

from app import pipeline
from app.db import get_db, init_db
from app.extraction import ParseError
from app.llm import LLMResponse
from app.ocr import TextResult

from .test_extraction import REWE, REWE_LLM


class FakeLLM:
    name = "fake"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    async def chat_json(self, system, user):
        self.calls.append(user)
        return LLMResponse(content=self.answers.pop(0))


@pytest.fixture
def receipt_id(monkeypatch):
    init_db()
    monkeypatch.setattr(pipeline, "extract_text", lambda path, mime: TextResult(REWE, "pdf", 1))
    with get_db() as db:
        return db.execute(
            "INSERT INTO receipts (file_path, mime_type, file_hash, status) "
            "VALUES ('x.pdf', 'application/pdf', 'h', 'verarbeitung')"
        ).lastrowid


def _load(rid):
    with get_db() as db:
        r = db.execute("SELECT * FROM receipts WHERE id = ?", (rid,)).fetchone()
        items = db.execute("SELECT name, price FROM items WHERE receipt_id = ? ORDER BY position", (rid,)).fetchall()
    return r, [(i["name"], i["price"]) for i in items]


def test_pipeline_saves_items_and_marks_done(receipt_id):
    llm = FakeLLM([json.dumps(REWE_LLM)])
    asyncio.run(pipeline.process_receipt(receipt_id, llm))
    r, items = _load(receipt_id)
    assert r["status"] == "fertig"
    assert r["merchant"] == "REWE" and r["purchase_date"] == "2026-11-03" and r["total"] == 12.28
    assert r["raw_text"] == REWE and r["text_source"] == "pdf"
    assert len(items) == 10 and items[0] == ("BANANE CHIQUITA", 1.49)
    assert "TSE" not in llm.calls[0]  # bereinigter Text geht an die KI


def test_pipeline_marks_review_on_mismatch(receipt_id):
    data = dict(REWE_LLM, positionen=REWE_LLM["positionen"][:3])
    asyncio.run(pipeline.process_receipt(receipt_id, FakeLLM([json.dumps(data)])))
    r, items = _load(receipt_id)
    assert r["status"] == "pruefen"
    assert "Summe der Positionen" in r["review_notes"]
    assert len(items) == 3  # wird trotzdem gespeichert, damit man korrigieren kann


def test_pipeline_retries_once_on_invalid_json(receipt_id):
    llm = FakeLLM(["Das ist kein JSON", json.dumps(REWE_LLM)])
    asyncio.run(pipeline.process_receipt(receipt_id, llm))
    r, _ = _load(receipt_id)
    assert r["status"] == "fertig"
    assert len(llm.calls) == 2 and "kein gültiges JSON" in llm.calls[1]
    assert "Versuch 1" in r["llm_raw"]


def test_pipeline_fails_after_two_invalid_answers(receipt_id):
    with pytest.raises(ParseError):
        asyncio.run(pipeline.process_receipt(receipt_id, FakeLLM(["nö", "immer noch nö"])))


def test_pipeline_rejects_empty_text(receipt_id, monkeypatch):
    monkeypatch.setattr(pipeline, "extract_text", lambda path, mime: TextResult("  \n ~~ \n", "ocr", 1))
    with pytest.raises(pipeline.NoTextError):
        asyncio.run(pipeline.process_receipt(receipt_id, FakeLLM([])))
