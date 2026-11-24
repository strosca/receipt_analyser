"""Regelbasiertes Auslesen von Positionen – ohne KI, für jeden Bon mit "Name … Preis"-Zeilen.

Das Ergebnis wird nur verwendet, wenn die Positionen exakt die Summenzeile ergeben
(siehe pipeline.py). Dann ist es praktisch bewiesen richtig und viel schneller als die KI.

Behandelt:
  BIO VOLLMILCH 3,8%                <- Name ohne Preis ...
    2 Stk x 1,29          2,58 A    <- ... Preis steht in der Mengenzeile darunter
  BANANEN                 1,87 A
    1,246 kg x 1,50 EUR/kg          <- Gewichtszeile ohne Endpreis: Zusatzinfo, überspringen
  2 x 1,25                          <- Mengenzeile VOR dem Artikel (dm, Aldi): überspringen
  Denkmit Tabs            2,50  1
  1   Arbeitszeit (Std.)   3,50   58,00 €   203,00 €   <- Rechnungszeile mit Pos.-Nr. und Spalten
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .extraction import NET_LINE, TOTAL_LINE, parse_amount

PRICE_AT_END = re.compile(
    r"(?P<price>-?\s?\d{1,3}(?:\.\d{3})*,\d{2})(?P<trail>-)?\s*(?:€|EUR)?\s*(?:[A-Z0-9]\*?|\*)?\s*$"
)
QTY_LINE = re.compile(
    r"^\s*\d+(?:,\d+)?\s*(?:Stk|St|Stück|kg|g|l|x)?\.?\s*[xX×*]\s*\d+,\d{2}", re.I
)
NOT_AN_ITEM = re.compile(
    r"^\s*(Zwischensumme|Summe|Gesamt|Total|Zu zahlen|MwSt|USt|Netto|Brutto|Steuer|Gegeben|Geg\.|Rückgeld|"
    r"Rueckgeld|Bar\b|EC-|Kartenzahlung|girocard|Betrag|Payback|Punkte)",
    re.I,
)
TRAILING_NUMBERS = re.compile(r"(\s+-?\d+(?:[.,]\d+)?\s*(?:€|EUR|Stk|St)?)+$")


@dataclass
class ParsedItem:
    name: str
    price: float


def _item_region(lines: list[str]) -> list[str]:
    """Alles vor der ersten Summen- bzw. Netto-Zeile."""
    for i, line in enumerate(lines):
        if TOTAL_LINE.match(line) or NET_LINE.search(line):
            return lines[:i]
    return lines


def _clean_name(text: str) -> str:
    text = text.strip()
    columns = [c for c in re.split(r"\s{2,}", text) if c.strip()]
    if len(columns) >= 2:
        if re.fullmatch(r"\d{1,3}\.?", columns[0].strip()):  # Positionsnummer
            columns = columns[1:]
        return columns[0].strip()
    text = re.sub(r"^\d{1,3}\.?\s+", "", text)  # Positionsnummer
    m = TRAILING_NUMBERS.search(text)
    if m:
        tail = m.group(0)
        # Nur abschneiden, wenn es eindeutig Mengen-/Preisspalten sind (nicht "PFAND 6 X 0,25")
        if "€" in tail or "EUR" in tail or len(re.findall(r"\d", tail)) and len(tail.split()) >= 2:
            text = text[: m.start()]
    return text.strip(" .:-")


def _price(line: str) -> tuple[float, str] | None:
    m = PRICE_AT_END.search(line)
    if not m:
        return None
    price = parse_amount(m.group("price").replace(" ", "") + (m.group("trail") or ""))
    if price is None:
        return None
    return price, line[: m.start()]


def _qty_price(line: str) -> float | None:
    """Endpreis einer Mengenzeile – nur wenn er HINTER "Menge x Einzelpreis" steht."""
    q = QTY_LINE.match(line)
    m = PRICE_AT_END.search(line)
    if not q or not m or m.start() < q.end():
        return None
    return parse_amount(m.group("price").replace(" ", "") + (m.group("trail") or ""))


def parse_items(raw_text: str) -> list[ParsedItem]:
    lines = [l.rstrip() for l in raw_text.splitlines() if l.strip()]
    items: list[ParsedItem] = []
    pending: str | None = None  # Name ohne Preis, wartet auf Mengenzeile
    for line in _item_region(lines):
        if NOT_AN_ITEM.match(line):
            pending = None
            continue
        if QTY_LINE.match(line):
            qty_price = _qty_price(line)
            if qty_price is not None and pending:
                items.append(ParsedItem(pending, qty_price))
            elif qty_price is not None and items and abs(items[-1].price - qty_price) > 0.005:
                # Artikelzeile zeigte den Einzelpreis, Mengenzeile den Gesamtpreis
                unit = parse_amount(re.search(r"[xX×*]\s*(\d+,\d{2})", line).group(1))
                if unit is not None and abs(items[-1].price - unit) < 0.005:
                    items[-1].price = qty_price
            pending = None
            continue
        found = _price(line)
        if found:
            name = _clean_name(found[1])
            if name and re.search(r"[A-Za-zÄÖÜäöü]{2}", name):
                items.append(ParsedItem(name, found[0]))
            pending = None
        else:
            text = line.strip()
            pending = text if re.search(r"[A-Za-zÄÖÜäöü]{2}", text) else None
    return items


def merge_quantity_lines(raw_text: str) -> str:
    """Für die KI: Mengenzeile mit Preis an den Namen darüber hängen, damit nichts verrutscht.

    "GUMMIBAERCHEN 200G" + "  3 Stk x 0,99   2,97 A"  ->  "GUMMIBAERCHEN 200G (3 Stk x 0,99)   2,97 A"
    """
    lines = raw_text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (
            line.strip()
            and not _price(line)
            and not QTY_LINE.match(line)
            and _qty_price(nxt) is not None
        ):
            m = PRICE_AT_END.search(nxt)
            qty = nxt[: m.start()].strip()
            out.append(f"{line.rstrip()} ({qty})   {m.group(0).strip()}")
            i += 2
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


# --------------------------------------------------------------------------- Kopfdaten ohne KI

HEADER_SKIP = re.compile(r"(\*|bon\b|beleg|testdaten|rechnung|kassenbon|willkommen|kundenbeleg|tel\.?|fax|www|@)", re.I)
PAYMENT_PATTERNS = [
    ("Karte", re.compile(r"girocard|EC[- ]?(Karte|Cash)|Maestro|V\s?PAY|Debit", re.I)),
    ("Kreditkarte", re.compile(r"\bVISA\b|Mastercard|Kreditkarte|American Express|AMEX", re.I)),
    ("PayPal", re.compile(r"PayPal", re.I)),
    ("Bar", re.compile(r"^\s*(Bar|BAR|Gegeben\s+Bar|Geg\.\s*Bar)\b|Rückgeld|Rueckgeld", re.M)),
]


def find_header_name(raw_text: str) -> str | None:
    """Ladenname aus dem Kopf eines Kassenbons, z. B. 'FRISCHMARKT MUSTERMANN'."""
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    if len(lines) < 5:  # kein richtiger Bon
        return None
    for line in lines[:6]:
        if HEADER_SKIP.search(line) or re.search(r"\d", line):
            continue
        if len(re.findall(r"[A-Za-zÄÖÜäöüß]", line)) >= 3:
            return re.sub(r"\s{2,}", " ", line)
    return None


def find_branch(raw_text: str) -> str | None:
    """Straße + PLZ/Ort aus dem Kopf, z. B. 'Musterstrasse 12, 12345 Musterstadt'."""
    lines = [re.sub(r"\s{2,}", " ", l.strip()) for l in raw_text.splitlines() if l.strip()][:10]
    for i, line in enumerate(lines):
        if re.match(r"^\d{5}\s+[A-ZÄÖÜa-zäöü]", line):
            prev = lines[i - 1] if i > 0 else ""
            if re.search(r"[A-Za-zäöüß]{3}.*\d", prev) and not HEADER_SKIP.search(prev):
                return f"{prev}, {line}"
            return line
    return None


def find_payment(raw_text: str) -> str | None:
    for name, pat in PAYMENT_PATTERNS:
        if pat.search(raw_text):
            return name
    return None
