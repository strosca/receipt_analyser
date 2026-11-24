import json
from pathlib import Path

import pytest

from app.extraction import (
    ParseError,
    build_result,
    canonical_merchant,
    clean_text,
    find_date,
    find_total,
    parse_amount,
    parse_date,
    parse_llm_response,
)

FIX = Path(__file__).parent / "fixtures"
REWE = (FIX / "rewe_ebon.txt").read_text(encoding="utf-8")
DM = (FIX / "dm_ocr.txt").read_text(encoding="utf-8")

REWE_LLM = {
    "haendler": "REWE Markt GmbH",
    "filiale": "Musterstraße 12, 12345 Musterstadt",
    "datum": "03.11.2026",
    "gesamtsumme": 12.28,
    "zahlungsart": "Karte",
    "positionen": [
        {"name": "BANANE CHIQUITA", "preis": 1.49},
        {"name": "VOLLMILCH 3,5%", "preis": 1.19},
        {"name": "MILKA ALPENMILCH", "preis": 2.58},
        {"name": "COCA COLA 1,0L", "preis": 1.29},
        {"name": "PFAND 0,25 EURO", "preis": 0.25},
        {"name": "HARIBO GOLDBAEREN", "preis": 0.99},
        {"name": "BIO EIER M 10ER", "preis": 3.49},
        {"name": "NIVEA DUSCHGEL", "preis": 1.95},
        {"name": "LEERGUT", "preis": -0.75},
        {"name": "RABATT HARIBO", "preis": -0.20},
    ],
}

DM_LLM = {
    "positionen": [
        {"name": "Balea Duschgel Fresh 300ml", "preis": 0.95},
        {"name": "alverde Shampoo Repair", "preis": 2.45},
        {"name": "Denkmit Spülmaschinentabs", "preis": 2.50},
        {"name": "dmBio Haferflocken 500g", "preis": 0.85},
        {"name": "Dontodent Zahnbürste", "preis": 1.45},
        {"name": "Coupon 10% auf Balea", "preis": -0.50},
    ]
}


# ---------------------------------------------------------------- Zahlen & Datum

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1,99", 1.99), ("1.234,56", 1234.56), ("-0,50", -0.50), ("0,50-", -0.50),
        ("2,99 €", 2.99), ("EUR 12,28", 12.28), ("1,49 B", 1.49), (3, 3.0), (1.5, 1.5),
        ("12.28", 12.28), ("1.234", 1234.0), ("abc", None), (None, None), ("", None),
    ],
)
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [("03.11.2026", "2026-11-03"), ("3.11.26", "2026-11-03"), ("2026-11-03", "2026-11-03"),
     ("Datum: 03.11.2026 18:42", "2026-11-03"), ("31.02.2026", None), ("", None), (None, None)],
)
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


# ---------------------------------------------------------------- Gegenprüfung im Text

def test_find_total_and_date_rewe():
    assert find_total(REWE) == 12.28  # nicht 11,22 aus der MwSt-Tabelle
    assert find_date(REWE) == "2026-11-03"


def test_find_total_and_date_dm():
    assert find_total(DM) == 7.70
    assert find_date(DM) == "2026-11-03"


def test_canonical_merchant():
    assert canonical_merchant("REWE Markt GmbH", REWE) == "REWE"
    assert canonical_merchant(None, DM) == "dm"
    assert canonical_merchant("dm-drogerie markt", "") == "dm"
    assert canonical_merchant("Bäckerei Schmidt", "") == "Bäckerei Schmidt"
    assert canonical_merchant(None, "irgendwas") is None


def test_clean_text_removes_noise_and_truncates():
    text, truncated = clean_text(REWE, 10_000)
    assert not truncated
    assert "TSE" not in text and "Signatur" not in text and "Vielen Dank" not in text
    assert "dGVzdHNpZ25hdHVy" not in text  # Base64-Zeile
    assert "BANANE CHIQUITA" in text and "SUMME" in text and "-0,75" in text
    short, truncated = clean_text(REWE, 100)
    assert truncated and len(short) == 100


# ---------------------------------------------------------------- KI-Antwort einlesen

def test_parse_clean_json():
    r = parse_llm_response(json.dumps(REWE_LLM))
    assert r.haendler == "REWE Markt GmbH"
    assert len(r.positionen) == 10
    assert r.positionen[-1].preis == -0.20


def test_parse_tolerates_markdown_strings_and_aliases():
    content = (
        "Hier ist das Ergebnis:\n```json\n"
        '{"merchant": "dm", "date": "03.11.26", "total": "7,70", "payment": null,\n'
        ' "items": [{"name": "Balea Duschgel", "price": "0,95"},\n'
        '           {"artikel": "Coupon", "betrag": "0,50-"},\n'
        '           {"name": "", "preis": 1},\n'
        '           {"name": "kaputt", "preis": "n/a"}]}\n```'
    )
    r = parse_llm_response(content)
    assert r.haendler == "dm" and r.gesamtsumme == 7.70 and r.datum == "03.11.26"
    assert [(i.name, i.preis) for i in r.positionen] == [("Balea Duschgel", 0.95), ("Coupon", -0.50)]


@pytest.mark.parametrize("bad", ["", "keine Ahnung", "[1, 2]", '{"positionen": "keine"}', "{kaputt"])
def test_parse_rejects_garbage(bad):
    with pytest.raises(ParseError):
        parse_llm_response(bad)


def test_null_strings_become_none():
    r = parse_llm_response('{"haendler": "null", "filiale": "", "positionen": []}')
    assert r.haendler is None and r.filiale is None


# ---------------------------------------------------------------- Plausibilität

def test_consistent_rewe_receipt_is_done():
    res = build_result(parse_llm_response(json.dumps(REWE_LLM)), REWE, 0.05)
    assert res.review_notes == []
    assert res.merchant == "REWE"
    assert res.purchase_date == "2026-11-03"
    assert res.total == 12.28 and res.items_sum == 12.28


def test_missing_item_triggers_review():
    data = dict(REWE_LLM, positionen=REWE_LLM["positionen"][:-1])  # Rabatt fehlt
    res = build_result(parse_llm_response(json.dumps(data)), REWE, 0.05)
    assert res.needs_review
    assert any("Summe der Positionen 12,48 €" in n and "12,28 €" in n for n in res.review_notes)


def test_small_rounding_difference_is_tolerated():
    data = dict(REWE_LLM, gesamtsumme=12.30)
    res = build_result(parse_llm_response(json.dumps(data)), REWE, 0.05)
    assert not res.needs_review


def test_regex_total_used_when_llm_total_missing_or_wrong():
    data = dict(REWE_LLM, gesamtsumme=None)
    assert build_result(parse_llm_response(json.dumps(data)), REWE, 0.05).total == 12.28
    # KI hat die Netto-Summe erwischt, Positionen passen aber zur SUMME-Zeile
    data = dict(REWE_LLM, gesamtsumme=11.22)
    res = build_result(parse_llm_response(json.dumps(data)), REWE, 0.05)
    assert res.total == 12.28 and not res.needs_review


def test_date_and_merchant_fallback_from_text():
    res = build_result(parse_llm_response(json.dumps(DM_LLM)), DM, 0.05)
    assert res.merchant == "dm" and res.purchase_date == "2026-11-03" and res.total == 7.70
    assert res.review_notes == []


def test_empty_result_lists_all_problems():
    res = build_result(parse_llm_response("{}"), "nur Müll", 0.05, truncated=True)
    joined = " ".join(res.review_notes)
    for expected in ("Keine Positionen", "Keine Gesamtsumme", "Kein Datum", "Kein Händler", "gekürzt"):
        assert expected in joined
