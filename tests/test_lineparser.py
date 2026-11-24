import asyncio
from pathlib import Path

import pytest

from app import pipeline
from app.lineparser import find_branch, find_header_name, find_payment, merge_quantity_lines, parse_items

from .test_extraction import DM, REWE
from .test_invoices import INVOICE

MUSTERBON = (Path(__file__).parent / "fixtures" / "musterbon.txt").read_text(encoding="utf-8")

MUSTERBON_ITEMS = [
    ("BIO VOLLMILCH 3,8%", 2.58), ("VOLLKORNBROT 750G", 2.49), ("BANANEN", 1.87),
    ("SCHOKORIEGEL 5ER", 2.19), ("GUMMIBAERCHEN 200G", 2.97), ("KARTOFFELCHIPS PAPRIKA", 1.79),
    ("MINERALWASSER 6X1,5L", 3.54), ("PFAND 6 X 0,25", 1.50), ("DUSCHGEL SENSITIV", 1.45),
    ("ZAHNPASTA", 1.90), ("SPUELMITTEL ZITRONE", 1.15), ("BIER PILS 0,5L", 3.56),
    ("PFAND 4 X 0,08", 0.32), ("RABATT SCHOKORIEGEL", -0.40), ("LEERGUT", -2.25),
]


def _items(text):
    return [(i.name, i.price) for i in parse_items(text)]


def test_musterbon_quantity_lines_below_item():
    assert _items(MUSTERBON) == MUSTERBON_ITEMS


def test_musterbon_without_layout_spacing():
    # So wie pdftotext ohne -layout bzw. OCR ohne Spaltenabstände
    flat = "\n".join(" ".join(l.split()) for l in MUSTERBON.splitlines())
    assert _items(flat) == MUSTERBON_ITEMS


def test_rewe_quantity_line_after_priced_item():
    items = _items(REWE)
    assert ("MILKA ALPENMILCH", 2.58) in items
    assert ("PFAND 0,25 EURO", 0.25) in items
    assert len(items) == 10 and round(sum(p for _, p in items), 2) == 12.28


def test_dm_quantity_line_before_item_and_tax_digit():
    items = _items(DM)
    assert ("Denkmit Spülmaschinentabs", 2.50) in items
    assert ("Coupon 10% auf Balea", -0.50) in items
    assert round(sum(p for _, p in items), 2) == 7.70


def test_invoice_columns_and_position_numbers():
    assert _items(INVOICE) == [
        ("Arbeitszeit Elektroinstallation (Std.)", 203.00),
        ("Steckdose 2-fach, Unterputz", 51.60),
        ("Installationskabel NYM-J 3x1,5 (m)", 22.25),
        ("Anfahrtspauschale", 35.00),
    ]


def test_invoice_without_layout_spacing():
    flat = "\n".join(" ".join(l.split()) for l in INVOICE.splitlines())
    assert [n for n, _ in _items(flat)] == [
        "Arbeitszeit Elektroinstallation (Std.)", "Steckdose 2-fach, Unterputz",
        "Installationskabel NYM-J 3x1,5 (m)", "Anfahrtspauschale",
    ]


def test_merge_quantity_lines_for_llm():
    merged = merge_quantity_lines(MUSTERBON)
    assert "GUMMIBAERCHEN 200G (3 Stk x 0,99)   2,97 A" in merged
    assert "BIO VOLLMILCH 3,8% (2 Stk x 1,29)   2,58 A" in merged
    assert "1,246 kg x 1,50 EUR/kg" in merged  # Gewichtszeile ohne Preis bleibt
    assert "\n      3 Stk x 0,99" not in merged


def test_header_branch_payment():
    assert find_header_name(MUSTERBON) == "FRISCHMARKT MUSTERMANN"
    assert find_branch(MUSTERBON) == "Musterstrasse 12, 12345 Musterstadt"
    assert find_payment(MUSTERBON) == "Karte"
    assert find_payment("Gegeben Bar 20,00\nRückgeld 2,00") == "Bar"


# ---------------------------------------------------------------- in der Pipeline

@pytest.mark.parametrize("text, merchant, total", [
    (MUSTERBON, "FRISCHMARKT MUSTERMANN", 24.66),
    (REWE, "REWE", 12.28),
    (DM, "dm", 7.70),
    (INVOICE, "Mustermann Elektrotechnik GmbH", 371.10),
])
def test_rules_succeed_without_llm(text, merchant, total):
    res = pipeline.try_rules(text)
    assert res is not None
    assert res.merchant == merchant and res.total == total and res.items_sum == total
    assert res.purchase_date is not None
    assert res.review_notes == []
    assert "ohne KI" in res.info_notes[0]


def test_rules_give_up_when_sum_does_not_match():
    broken = MUSTERBON.replace("LEERGUT                                      -2,25 B\n", "")
    assert pipeline.try_rules(broken) is None


def test_rules_can_be_disabled(monkeypatch):
    monkeypatch.setenv("RULE_PARSER", "false")
    from app.config import get_settings
    get_settings.cache_clear()
    assert pipeline.try_rules(MUSTERBON) is None
