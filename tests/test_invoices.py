"""Rechnungen (statt Kassenbons): Netto-Positionen mit MwSt-Aufschlag, Absender vs. Empfänger."""
import json
from pathlib import Path

from app.extraction import (
    build_result,
    canonical_merchant,
    find_sender,
    find_sender_line,
    find_total,
    gross_up,
    is_net_invoice,
    parse_llm_response,
)

from .test_extraction import DM, REWE

INVOICE = (Path(__file__).parent / "fixtures" / "rechnung_netto.txt").read_text(encoding="utf-8")

# So hat die KI die Musterrechnung tatsächlich beantwortet: Netto-Positionen, Brutto-Summe
INVOICE_LLM = {
    "haendler": "Mustermann Elektrotechnik GmbH",
    "filiale": "Musterweg 7, Musterstadt",
    "datum": "24.11.2026",
    "gesamtsumme": 371.10,
    "positionen": [
        {"name": "Arbeitszeit Elektroinstallation (Std.)", "preis": 203.00},
        {"name": "Steckdose 2-fach, Unterputz", "preis": 51.60},
        {"name": "Installationskabel NYM-J 3x1,5 (m)", "preis": 22.25},
        {"name": "Anfahrtspauschale", "preis": 35.00},
    ],
}


def _result(llm: dict, text: str = INVOICE, own=None):
    return build_result(parse_llm_response(json.dumps(llm)), text, 0.05, own_names=own)


# ---------------------------------------------------------------- Netto-Rechnungen

def test_net_invoice_is_detected():
    assert find_total(INVOICE) == 371.10
    assert is_net_invoice(311.85, 371.10, INVOICE, 0.05)


def test_supermarket_receipts_are_not_net_invoices():
    # REWE hat eine MwSt-Tabelle, aber die Positionen sind schon brutto
    assert not is_net_invoice(12.28, 12.28, REWE, 0.05)
    # Fehlt eine Position, darf das NICHT als Netto-Rechnung "wegerklärt" werden
    assert not is_net_invoice(11.84, 12.28, REWE, 0.05)  # 0,44 = Steuerbetrag in der Tabelle
    assert not is_net_invoice(7.20, 7.70, DM, 0.05)


def test_net_invoice_items_are_grossed_up():
    res = _result(INVOICE_LLM)
    assert res.review_notes == []
    assert res.total == 371.10 and res.items_sum == 371.10
    prices = dict(res.items)
    assert prices["Arbeitszeit Elektroinstallation (Std.)"] == 241.57  # 203,00 × 1,19
    assert prices["Anfahrtspauschale"] == 41.65
    assert "Netto-Rechnung" in res.info_notes[0]


def test_net_invoice_when_llm_returns_net_total():
    res = _result(dict(INVOICE_LLM, gesamtsumme=311.85))
    assert res.total == 371.10 and res.items_sum == 371.10 and res.review_notes == []


def test_gross_up_keeps_exact_total_with_rounding():
    items = gross_up([("a", 0.33), ("b", 0.33), ("c", 0.34)], 1.19)
    assert round(sum(p for _, p in items), 2) == 1.19


# ---------------------------------------------------------------- Absender vs. Empfänger

def test_sender_line_is_found():
    assert find_sender_line(INVOICE) == "Mustermann Elektrotechnik GmbH"
    assert find_sender(INVOICE) == "Mustermann Elektrotechnik GmbH"
    assert find_sender_line(REWE) is None


def test_recipient_from_llm_is_replaced_by_sender():
    for wrong in ("Max Beispiel", "Beispielallee 3"):
        res = _result(dict(INVOICE_LLM, haendler=wrong, filiale="Beispielallee 3, 54321 Beispielhausen"))
        assert res.merchant == "Mustermann Elektrotechnik GmbH", wrong
        assert res.branch is None


def test_own_names_are_never_the_merchant():
    text = "Sehr geehrte Damen und Herren\nRechnung an Jakob Muster\nSUMME 10,00\nHuber Elektro GmbH\n"
    assert canonical_merchant("Jakob Muster", text, ["Jakob Muster"]) == "Huber Elektro GmbH"
    assert canonical_merchant("jakob muster", "", ["Jakob Muster"]) is None


def test_letter_with_salutation_block():
    text = (
        "Schreinerei Holz & Co. KG\n"
        "Herr\nJakob Muster\nGartenweg 2\n99999 Irgendwo\n"
        "Rechnung Nr. 5 vom 01.10.2026\nTisch  100,00\nGesamtbetrag 100,00\n"
    )
    assert canonical_merchant("Jakob Muster", text) == "Schreinerei Holz & Co. KG"


def test_sender_line_beats_wrong_llm_value():
    res = _result(dict(INVOICE_LLM, haendler="Musterbank"))
    assert res.merchant == "Mustermann Elektrotechnik GmbH"
