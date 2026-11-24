"""Bontext für die KI vorbereiten, KI-Antwort auswerten und auf Plausibilität prüfen.

Dieses Modul ist bewusst frei von Datenbank- und Netzwerkcode, damit es sich gut testen lässt.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, Field, ValidationError, field_validator

# --------------------------------------------------------------------------- Prompt

SYSTEM_PROMPT = """Du liest deutsche Kassenbons und Rechnungen aus und antwortest NUR mit JSON.

Antwortformat:
{"haendler": "REWE", "filiale": "Hauptstr. 5, Berlin", "datum": "TT.MM.JJJJ", "gesamtsumme": 12.34,
 "zahlungsart": "Karte", "positionen": [{"name": "BANANEN", "preis": 1.99}]}

Regeln:
- "haendler": wer den Bon bzw. die Rechnung AUSGESTELLT hat (z. B. REWE, EDEKA, Lidl, Aldi, dm, Rossmann, Amazon, eine Firma).
  Bei Rechnungen und Briefen ist das der Absender: Briefkopf ganz oben, kleine Absenderzeile über der Anschrift, Fußzeile mit GmbH/Bankverbindung.
  NIEMALS der Empfänger bzw. Kunde (die Anschrift mit "Herr"/"Frau", Rechnungs- oder Lieferadresse).
- "filiale": Adresse oder Ort des Ausstellers, sonst null.
- "datum": Kaufdatum bzw. Rechnungsdatum als TT.MM.JJJJ, sonst null.
- "gesamtsumme": der zu zahlende Endbetrag (SUMME, Gesamt, Zu zahlen, Gesamtbetrag, Rechnungsbetrag, Total), bei Rechnungen der Brutto-Betrag.
- "zahlungsart": "Bar", "Karte", "PayPal", "Überweisung" o. Ä., sonst null.
- "positionen": JEDE gekaufte Position in der Reihenfolge des Bons.
  - "name": genau wie auf dem Bon, ohne Preis, ohne Positionsnummer.
  - "preis": Gesamtpreis der Zeile (der Betrag ganz rechts bzw. Spalte "Gesamt"). Bei "2 x 0,99" ist der Preis 1.98.
  - Stehen die Positionen netto da und die MwSt wird erst darunter addiert, übernimm die Preise trotzdem genau so wie gedruckt.
  - Mengen- oder Gewichtszeilen (z. B. "2 Stk x 0,99" oder "0,456 kg x 2,99 EUR/kg") gehören zur Position darüber und sind KEINE eigene Position.
  - Pfand ist eine eigene Position mit positivem Preis.
  - Leergut, Rabatte, Coupons und Preisnachlässe sind eigene Positionen mit NEGATIVEM Preis.
- Ignoriere: MwSt-/Steuerzeilen, Netto/Brutto, Zwischensummen, Gegeben, Rückgeld, Kartenzahlungsdetails, Bonus-/Payback-Punkte, Werbung.
- Zahlen im JSON mit Punkt als Dezimaltrennzeichen (1,99 auf dem Bon -> 1.99).
- Die Summe aller Positionen muss die Gesamtsumme ergeben.
- Erfinde nichts. Fehlt etwas, setze null."""

USER_PROMPT = "Bontext:\n```\n{text}\n```\nAntworte nur mit dem JSON-Objekt."

RETRY_PROMPT = (
    "Deine letzte Antwort war kein gültiges JSON im verlangten Format ({error}). "
    "Antworte jetzt NUR mit dem JSON-Objekt."
)

# Zeilen ohne Nutzen für die Auswertung (TSE-Signaturen, Kassendaten), spart Kontext
NOISE_LINE = re.compile(
    r"(TSE|Signatur|Seriennummer|Serien-Nr|Transaktions?(nummer|-Nr)|Prüfwert|Zeitformat|Hashalgorithmus|"
    r"Public Key|Kassen-?ID|Terminal-?ID|Trace-?Nr|Beleg-?Nr\.?\s*\d|Genehmigungs|AID\s|EMV|"
    r"Vielen Dank|Danke für|Öffnungszeiten|www\.|https?://|Steuer-?Nr|USt-?Id)",
    re.IGNORECASE,
)
BASE64_LINE = re.compile(r"^[A-Za-z0-9+/=]{24,}$")


def clean_text(raw: str, max_chars: int) -> tuple[str, bool]:
    """Kompakter Text für die KI. Gibt (Text, wurde_gekürzt) zurück."""
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"[ \t]{3,}", "   ", line.rstrip())
        stripped = line.strip()
        if not stripped or len(stripped) < 2 and not stripped.isalnum():
            continue
        if NOISE_LINE.search(stripped) or BASE64_LINE.match(stripped.replace(" ", "")):
            continue
        if re.fullmatch(r"[-=_*#.~ ]+", stripped):  # Trennlinien
            continue
        lines.append(line)
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


# --------------------------------------------------------------------------- Zahlen & Datum

AMOUNT_RE = re.compile(r"^(?P<sign>[-−])?\s*(?P<num>\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*(?P<trail>[-−])?$")


def parse_amount(value) -> float | None:
    """Wandelt '1,99', '1.234,56', '-0,50', '0,50-', '2,99 €', 3 in float um."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    s = str(value).strip().replace("€", "").replace("EUR", "").replace(" ", " ").strip()
    s = re.sub(r"\s*[A-Z]$", "", s)  # Steuerkennzeichen am Ende, z. B. "1,99 A"
    m = AMOUNT_RE.match(s)
    if not m:
        return None
    num = m.group("num").replace(" ", "")
    if "," in num:
        num = num.replace(".", "").replace(",", ".")
    elif num.count(".") == 1 and len(num.split(".")[1]) == 3:
        num = num.replace(".", "")  # "1.234" = Tausender
    try:
        result = float(num)
    except ValueError:
        return None
    if m.group("sign") or m.group("trail"):
        result = -result
    return round(result, 2)


DATE_PATTERNS = [
    re.compile(r"\b(?P<d>\d{1,2})\.(?P<m>\d{1,2})\.(?P<y>\d{4}|\d{2})\b"),
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b"),
    re.compile(r"\b(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4}|\d{2})\b"),
]


def parse_date(value) -> str | None:
    """Gibt ein ISO-Datum (JJJJ-MM-TT) zurück oder None."""
    if not value:
        return None
    for pat in DATE_PATTERNS:
        m = pat.search(str(value))
        if not m:
            continue
        d, mo, y = int(m.group("d")), int(m.group("m")), int(m.group("y"))
        if y < 100:
            y += 2000
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- Händler

KNOWN_MERCHANTS: list[tuple[str, re.Pattern]] = [
    ("REWE", re.compile(r"\bREWE\b", re.I)),
    ("EDEKA", re.compile(r"\bE\s?DEKA\b|\bE\s?center\b|Marktkauf", re.I)),
    ("Lidl", re.compile(r"\bLIDL\b", re.I)),
    ("Aldi", re.compile(r"\bALDI\b", re.I)),
    ("Rossmann", re.compile(r"\bROSSMANN\b", re.I)),
    ("dm", re.compile(r"\bdm[- ]?drogerie|\bdm[- ]markt\b|^\s*dm\s*$", re.I | re.M)),
    ("Netto", re.compile(r"\bNETTO\b", re.I)),
    ("Penny", re.compile(r"\bPENNY\b", re.I)),
    ("Kaufland", re.compile(r"\bKAUFLAND\b", re.I)),
    ("Norma", re.compile(r"\bNORMA\b", re.I)),
    ("Müller", re.compile(r"\bM(Ü|UE)LLER\b", re.I)),
    ("Amazon", re.compile(r"\bAMAZON\b", re.I)),
    ("Globus", re.compile(r"\bGLOBUS\b", re.I)),
    ("tegut", re.compile(r"\bTEGUT\b", re.I)),
]


LEGAL_FORM = re.compile(r"\b(GmbH|AG|UG|KG|OHG|GbR|e\.\s?K\.|e\.\s?V\.|SE|Ltd\.?|Inc\.?|S\.à r\.l\.|B\.V\.)(\s*&\s*Co\.?\s*(KG|OHG))?\b")
PLZ = re.compile(r"\b\d{5}\b")
SENDER_SEPARATORS = re.compile(r"\s+[·•|–-]\s+|\s*,\s+")
RECIPIENT_SALUTATION = re.compile(r"^\s*(Herr|Frau|Familie|Firma)\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]", "", s.lower())


def _lines(raw_text: str) -> list[str]:
    return [re.sub(r"\s{2,}", "  ", l).strip() for l in raw_text.splitlines() if l.strip()]


def find_sender_line(raw_text: str) -> str | None:
    """DIN-5008-Absenderzeile über dem Adressfeld: "Firma · Straße · 12345 Ort" -> "Firma"."""
    for line in _lines(raw_text)[:25]:
        left = line.split("  ")[0]  # bei Layout-Text steht rechts oft etwas anderes
        if "@" in left or "www." in left:
            continue  # Kontaktzeile im Briefkopf, nicht die Absenderzeile
        parts = [p.strip() for p in SENDER_SEPARATORS.split(left) if p.strip()]
        if (
            len(parts) >= 3
            and PLZ.search(parts[-1])
            and re.search(r"[A-Za-zÄÖÜäöü]{3}", parts[0])
            and not re.search(r"\d", parts[0])
        ):
            return parts[0]
    return None


def find_sender(raw_text: str) -> str | None:
    """Sucht den Aussteller einer Rechnung im Text, ohne KI.

    1. DIN-5008-Absenderzeile über dem Adressfeld
    2. Erste Zeile im oberen Bereich mit Rechtsform (GmbH, AG, …)
    """
    sender = find_sender_line(raw_text)
    if sender:
        return sender
    for line in _lines(raw_text)[:12]:
        left = line.split("  ")[0]
        m = LEGAL_FORM.search(left)
        if m and len(left) <= 80:
            return left[: m.end()].strip()
    return None


def recipient_lines(raw_text: str) -> list[str]:
    """Zeilen, die vermutlich zur Empfänger-Anschrift gehören (nach Absenderzeile bzw. 'Herr/Frau')."""
    lines = _lines(raw_text)[:30]
    sender_line = find_sender_line(raw_text)
    out = []
    for i, line in enumerate(lines):
        left = line.split("  ")[0]
        if RECIPIENT_SALUTATION.match(left):
            out.extend(l.split("  ")[0] for l in lines[i: i + 3])
        elif sender_line and left.startswith(sender_line) and PLZ.search(left):
            out.extend(l.split("  ")[0] for l in lines[i + 1: i + 4])
    return out


def is_own_or_recipient(value: str, raw_text: str, own_names: list[str]) -> bool:
    v = _norm(value)
    if not v:
        return False
    for own in own_names:
        o = _norm(own)
        if o and (o in v or v in o):
            return True
    for line in recipient_lines(raw_text):
        n = _norm(re.sub(RECIPIENT_SALUTATION, "", line))
        if n and len(n) > 5 and (v in n or n in v):
            return True
    return False


def canonical_merchant(llm_value: str | None, raw_text: str, own_names: list[str] | None = None) -> str | None:
    """Händler bestimmen und vereinheitlichen ('REWE Markt GmbH' und 'Rewe' -> 'REWE').

    Der Wert der KI wird verworfen, wenn er der eigene Name bzw. die Empfänger-Anschrift ist.
    """
    own_names = own_names or []
    if llm_value and is_own_or_recipient(llm_value, raw_text, own_names):
        llm_value = None
    if llm_value:
        for name, pat in KNOWN_MERCHANTS:
            if pat.search(llm_value) or llm_value.strip().lower() == name.lower():
                return name
    head = "\n".join(raw_text.splitlines()[:15])
    for name, pat in KNOWN_MERCHANTS:
        if pat.search(head):
            return name
    # Eine Absenderzeile über dem Adressfeld ist verlässlicher als die KI
    sender_line = find_sender_line(raw_text)
    if sender_line and not is_own_or_recipient(sender_line, raw_text, own_names):
        if llm_value and _norm(llm_value) and (_norm(llm_value) in _norm(sender_line) or _norm(sender_line) in _norm(llm_value)):
            return llm_value.strip()
        return sender_line
    from .lineparser import find_header_name  # spät importieren (zirkulärer Import)

    header = find_header_name(raw_text)
    if header and is_own_or_recipient(header, raw_text, own_names):
        header = None
    if llm_value and llm_value.strip():
        # KI hat nur einen Teil erwischt ("MUSTERMANN" statt "FRISCHMARKT MUSTERMANN")
        if header and _norm(llm_value) and _norm(llm_value) in _norm(header):
            return header
        return llm_value.strip()
    sender = find_sender(raw_text)
    if sender and not is_own_or_recipient(sender, raw_text, own_names):
        return sender
    return header


# --------------------------------------------------------------------------- Gegenprüfung per Regex

TOTAL_LINE = re.compile(
    r"^\s*(SUMME|Summe|GESAMT|Gesamt(betrag|summe)?|Zu zahlen|ZU ZAHLEN|Endbetrag|Rechnungsbetrag|"
    r"Gesamtpreis|TOTAL|Total|Betrag)\b[^\n\d-]*?(?P<amount>-?\d{1,3}(?:\.\d{3})*,\d{2})",
    re.M,
)


def find_total(raw_text: str) -> float | None:
    """Sucht die Summenzeile direkt im Text (unabhängig von der KI)."""
    matches = [parse_amount(m.group("amount")) for m in TOTAL_LINE.finditer(raw_text)]
    matches = [m for m in matches if m is not None]
    # Die erste SUMME-Zeile ist auf deutschen Bons meist die richtige (danach folgen MwSt usw.)
    return matches[0] if matches else None


AMOUNT_IN_TEXT = re.compile(r"(?<![\d,.])-?\d{1,3}(?:\.\d{3})*,\d{2}(?!\s*%)(?![\d])")
NET_LINE = re.compile(r"\b(Netto(betrag|summe)?|Summe\s+netto|Zwischensumme(\s+netto)?|Warenwert\s+netto|Gesamt\s+netto)\b", re.I)
TAX_ADD_LINE = re.compile(r"\b(zzgl\.?|zuzügl(ich|\.)|\+)\s*.*\b(MwSt|USt|Umsatzsteuer|Mehrwertsteuer)", re.I)
TAX_LINE = re.compile(r"\b(MwSt|USt|Umsatzsteuer|Mehrwertsteuer)\b(?!-?Id)", re.I)


def _amounts(line: str) -> list[float]:
    return [a for a in (parse_amount(m.group(0)) for m in AMOUNT_IN_TEXT.finditer(line)) if a is not None]


def is_net_invoice(items_sum: float, total: float | None, raw_text: str, tolerance: float) -> bool:
    """Erkennt Rechnungen, deren Positionen netto sind und bei denen die MwSt erst unten aufgeschlagen wird.

    Nötig sind zwei Belege im Text: eine Netto-Zeile mit genau der Summe der Positionen ODER
    eine "zzgl. MwSt"-Zeile, deren Betrag genau die Lücke zur Gesamtsumme erklärt.
    """
    if total is None or items_sum <= 0 or total - items_sum <= tolerance:
        return False
    gap = total - items_sum
    tax_amounts = []
    for line in raw_text.splitlines():
        if NET_LINE.search(line) and any(abs(a - items_sum) <= tolerance for a in _amounts(line)):
            return True
        if TAX_ADD_LINE.search(line) or TAX_LINE.search(line):
            tax_amounts.extend(_amounts(line))
            if TAX_ADD_LINE.search(line) and any(abs(a - gap) <= tolerance for a in _amounts(line)):
                return True
    # mehrere Steuersätze, z. B. "zzgl. 7 % MwSt 1,23" und "zzgl. 19 % MwSt 4,56"
    return len(tax_amounts) >= 2 and abs(sum(set(tax_amounts)) - gap) <= tolerance and bool(TAX_ADD_LINE.search(raw_text))


def gross_up(items: list[tuple[str, float]], total: float) -> list[tuple[str, float]]:
    """Rechnet Netto-Positionen anteilig auf brutto hoch, sodass die Summe genau stimmt."""
    net_sum = sum(p for _, p in items)
    factor = total / net_sum
    out = [(n, round(p * factor, 2)) for n, p in items]
    rest = round(total - sum(p for _, p in out), 2)
    if rest:
        i = max(range(len(out)), key=lambda k: abs(out[k][1]))
        out[i] = (out[i][0], round(out[i][1] + rest, 2))
    return out


def find_date(raw_text: str) -> str | None:
    for line in raw_text.splitlines():
        iso = parse_date(line)
        if iso and 2000 <= int(iso[:4]) <= date.today().year + 1:
            return iso
    return None


# --------------------------------------------------------------------------- KI-Antwort


class LLMItem(BaseModel):
    name: str = Field(min_length=1)
    preis: float

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, v):
        return str(v).strip() if v is not None else v

    @field_validator("preis", mode="before")
    @classmethod
    def _preis(cls, v):
        amount = parse_amount(v)
        if amount is None:
            raise ValueError(f"ungültiger Preis: {v!r}")
        return amount


class LLMReceipt(BaseModel):
    haendler: str | None = None
    filiale: str | None = None
    datum: str | None = None
    gesamtsumme: float | None = None
    zahlungsart: str | None = None
    positionen: list[LLMItem] = Field(default_factory=list)

    @field_validator("gesamtsumme", mode="before")
    @classmethod
    def _summe(cls, v):
        return parse_amount(v)

    @field_validator("haendler", "filiale", "zahlungsart", "datum", mode="before")
    @classmethod
    def _text(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        return s if s and s.lower() not in ("null", "none", "-", "unbekannt") else None


# Kleine Modelle weichen gern auf englische oder andere Schlüssel aus
KEY_ALIASES = {
    "haendler": ["händler", "merchant", "store", "shop", "laden", "geschaeft", "geschäft"],
    "filiale": ["branch", "address", "adresse", "location"],
    "datum": ["date", "kaufdatum"],
    "gesamtsumme": ["summe", "total", "gesamt", "betrag", "total_amount", "zu_zahlen"],
    "zahlungsart": ["payment", "payment_method", "zahlung"],
    "positionen": ["items", "artikel", "position", "products", "produkte"],
}
ITEM_ALIASES = {
    "name": ["artikel", "bezeichnung", "item", "description", "produkt"],
    "preis": ["price", "gesamtpreis", "total", "betrag", "summe", "amount"],
}


def _apply_aliases(d: dict, aliases: dict) -> dict:
    lower = {str(k).lower(): v for k, v in d.items()}
    out = {}
    for key, alts in aliases.items():
        for k in [key, *alts]:
            if k in lower and lower[k] is not None:
                out[key] = lower[k]
                break
    return out


class ParseError(Exception):
    pass


def parse_llm_response(content: str) -> LLMReceipt:
    """Liest die KI-Antwort tolerant ein (Markdown-Codeblöcke, Aliasse, Zahlen als Text)."""
    text = content.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ParseError("Antwort enthält kein JSON")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise ParseError(f"Ungültiges JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ParseError("JSON ist kein Objekt")

    data = _apply_aliases(data, KEY_ALIASES)
    items_raw = data.get("positionen") or []
    if not isinstance(items_raw, list):
        raise ParseError("'positionen' ist keine Liste")
    items = []
    for it in items_raw:
        if not isinstance(it, dict):
            continue
        it = _apply_aliases(it, ITEM_ALIASES)
        try:
            items.append(LLMItem(**it))
        except ValidationError:
            continue  # unbrauchbare Zeile auslassen, fällt bei der Summenprüfung auf
    data["positionen"] = items
    try:
        return LLMReceipt(**data)
    except ValidationError as exc:
        raise ParseError(f"Unerwartete Struktur: {exc.errors()[0]['msg']}") from exc


# --------------------------------------------------------------------------- Ergebnis & Plausibilität


@dataclass
class ReceiptResult:
    merchant: str | None
    branch: str | None
    purchase_date: str | None
    total: float | None
    payment_method: str | None
    items: list[tuple[str, float]]
    review_notes: list[str] = field(default_factory=list)
    info_notes: list[str] = field(default_factory=list)  # Hinweise ohne Prüfbedarf

    @property
    def needs_review(self) -> bool:
        return bool(self.review_notes)

    @property
    def items_sum(self) -> float:
        return round(sum(p for _, p in self.items), 2)


def _fmt(v: float) -> str:
    return f"{v:.2f}".replace(".", ",") + " €"


def build_result(
    parsed: LLMReceipt,
    raw_text: str,
    tolerance: float,
    truncated: bool = False,
    own_names: list[str] | None = None,
) -> ReceiptResult:
    """Kombiniert KI-Antwort und Regex-Gegenprüfung und sammelt Gründe für 'prüfen'."""
    notes: list[str] = []
    info: list[str] = []
    items = [(i.name, i.preis) for i in parsed.positionen if i.name]
    items_sum = round(sum(p for _, p in items), 2)

    llm_total = parsed.gesamtsumme
    regex_total = find_total(raw_text)
    total = llm_total
    if total is None:
        total = regex_total
    elif regex_total is not None and abs(regex_total - llm_total) > tolerance:
        # Widerspruch: den Wert nehmen, zu dem die Positionen passen
        if abs(items_sum - regex_total) <= tolerance or is_net_invoice(items_sum, regex_total, raw_text, tolerance):
            total = regex_total
        elif abs(items_sum - llm_total) > tolerance:
            notes.append(f"KI-Summe {_fmt(llm_total)} weicht von der Summenzeile {_fmt(regex_total)} ab.")

    # Rechnung mit Netto-Positionen: auf brutto hochrechnen, damit die Auswertung echte Ausgaben zeigt
    if items and total is not None and abs(items_sum - total) > tolerance and is_net_invoice(items_sum, total, raw_text, tolerance):
        items = gross_up(items, total)
        info.append(
            f"Netto-Rechnung: Positionen wurden inkl. MwSt umgerechnet ({_fmt(items_sum)} netto → {_fmt(total)} brutto)."
        )
        items_sum = round(sum(p for _, p in items), 2)

    purchase_date = parse_date(parsed.datum) or find_date(raw_text)
    if purchase_date and not (2000 <= int(purchase_date[:4]) <= date.today().year + 1):
        notes.append(f"Datum {purchase_date} ist unplausibel.")
        purchase_date = None

    merchant = canonical_merchant(parsed.haendler, raw_text, own_names)

    if not items:
        notes.append("Keine Positionen erkannt.")
    if total is None:
        notes.append("Keine Gesamtsumme gefunden.")
    elif items and abs(items_sum - total) > tolerance:
        notes.append(
            f"Summe der Positionen {_fmt(items_sum)} ≠ Gesamtsumme {_fmt(total)} "
            f"(Differenz {_fmt(round(total - items_sum, 2))})."
        )
    if not purchase_date:
        notes.append("Kein Datum erkannt.")
    if not merchant:
        notes.append("Kein Händler erkannt.")
    if truncated:
        notes.append("Bontext war zu lang und wurde für die KI gekürzt – evtl. fehlen Positionen.")

    branch = parsed.filiale
    if branch and is_own_or_recipient(branch, raw_text, own_names or []):
        branch = None  # KI hat die Empfänger-Anschrift erwischt

    return ReceiptResult(
        merchant=merchant,
        branch=branch,
        purchase_date=purchase_date,
        total=total,
        payment_method=parsed.zahlungsart,
        items=items,
        review_notes=notes,
        info_notes=info,
    )
