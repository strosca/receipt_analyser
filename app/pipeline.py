"""Verarbeitung eines Bons: Text gewinnen -> KI-Auswertung -> Prüfung -> speichern."""
import asyncio
import logging

from .config import get_settings
from .db import STATUS_DONE, STATUS_REVIEW, get_db
from .extraction import (
    RETRY_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
    LLMItem,
    LLMReceipt,
    ParseError,
    ReceiptResult,
    build_result,
    clean_text,
    find_total,
    is_net_invoice,
    parse_llm_response,
)
from .lineparser import find_branch, find_payment, merge_quantity_lines, parse_items
from .llm import LLMClient, get_llm, llm_lock
from .ocr import extract_text
from .storage import receipt_file_path
from .worker import set_progress

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 20


class NoTextError(Exception):
    pass


async def ask_llm(llm: LLMClient, text: str) -> tuple[LLMReceipt, str]:
    """Fragt die KI. Bei ungültigem JSON genau ein zweiter Versuch. Gibt (Ergebnis, Rohantworten) zurück."""
    user = USER_PROMPT.format(text=text)
    resp = await llm.chat_json(SYSTEM_PROMPT, user)
    try:
        return parse_llm_response(resp.content), resp.content
    except ParseError as exc:
        log.warning("KI-Antwort unbrauchbar (%s), zweiter Versuch", exc)
        first = resp.content
        resp = await llm.chat_json(SYSTEM_PROMPT, user + "\n\n" + RETRY_PROMPT.format(error=exc))
        raw = f"--- Versuch 1 ---\n{first}\n--- Versuch 2 ---\n{resp.content}"
        try:
            return parse_llm_response(resp.content), raw
        except ParseError as exc2:
            raise ParseError(f"KI lieferte zweimal kein gültiges JSON: {exc2}. Antwort: {resp.content[:300]}") from exc2


def try_rules(raw_text: str) -> ReceiptResult | None:
    """Positionen ohne KI auslesen. Nur gültig, wenn sie exakt die Summenzeile ergeben."""
    settings = get_settings()
    if not settings.rule_parser:
        return None
    items = parse_items(raw_text)
    total = find_total(raw_text)
    if not items or total is None:
        return None
    items_sum = round(sum(i.price for i in items), 2)
    tol = settings.sum_tolerance
    if abs(items_sum - total) > tol and not is_net_invoice(items_sum, total, raw_text, tol):
        log.info("Regeln: %d Positionen = %.2f, Summe %.2f -> passt nicht, frage KI", len(items), items_sum, total)
        return None
    parsed = LLMReceipt(
        filiale=find_branch(raw_text),
        gesamtsumme=total,
        zahlungsart=find_payment(raw_text),
        positionen=[LLMItem(name=i.name, preis=i.price) for i in items],
    )
    result = build_result(parsed, raw_text, tol, own_names=settings.own_names_list)
    result.info_notes.insert(0, "Positionen per Regel erkannt (ohne KI).")
    return result


def save_result(receipt_id: int, result: ReceiptResult, llm_raw: str | None) -> None:
    status = STATUS_REVIEW if result.needs_review else STATUS_DONE
    with get_db() as db:
        db.execute(
            """UPDATE receipts SET merchant = ?, branch = ?, purchase_date = ?, total = ?, payment_method = ?,
                   llm_raw = ?, review_notes = ?, status = ?, status_detail = ?,
                   processed_at = datetime('now'), updated_at = datetime('now')
               WHERE id = ?""",
            (
                result.merchant, result.branch, result.purchase_date, result.total, result.payment_method,
                llm_raw, "\n".join(result.review_notes) or None, status,
                " ".join(result.info_notes) or None, receipt_id,
            ),
        )
        db.execute("DELETE FROM items WHERE receipt_id = ?", (receipt_id,))
        db.executemany(
            "INSERT INTO items (receipt_id, position, name, price) VALUES (?, ?, ?, ?)",
            [(receipt_id, i, name, price) for i, (name, price) in enumerate(result.items, 1)],
        )


async def process_receipt(receipt_id: int, llm: LLMClient | None = None) -> None:
    settings = get_settings()
    with get_db() as db:
        r = db.execute("SELECT file_path, mime_type FROM receipts WHERE id = ?", (receipt_id,)).fetchone()

    # 1. Text gewinnen
    set_progress(receipt_id, "Texterkennung läuft")
    text_result = await asyncio.to_thread(extract_text, receipt_file_path(r["file_path"]), r["mime_type"])
    log.info("Bon %s: %d Zeichen (%s, %d Seite/n)", receipt_id, len(text_result.text), text_result.source, text_result.pages)
    with get_db() as db:
        db.execute(
            "UPDATE receipts SET raw_text = ?, text_source = ?, updated_at = datetime('now') WHERE id = ?",
            (text_result.text, text_result.source, receipt_id),
        )
    prompt_text, truncated = clean_text(merge_quantity_lines(text_result.text), settings.llm_max_input_chars)
    if len(prompt_text.strip()) < MIN_TEXT_CHARS:
        raise NoTextError("Kein lesbarer Text erkannt. Foto evtl. unscharf oder zu dunkel.")

    # 2a. Schneller Weg: Positionen per Regel, wenn sie exakt die Summe ergeben
    result = try_rules(text_result.text)
    if result is not None:
        save_result(receipt_id, result, None)
        log.info("Bon %s: per Regel erkannt (%s, %d Positionen, Summe %s)",
                 receipt_id, result.merchant, len(result.items), result.total)
        return

    # 2b. KI-Auswertung (immer nur eine gleichzeitig)
    llm = llm or get_llm()
    set_progress(receipt_id, "Wartet auf KI")
    async with llm_lock:
        set_progress(receipt_id, "KI-Auswertung läuft (kann einige Minuten dauern)")
        parsed, llm_raw = await ask_llm(llm, prompt_text)

    # 3. Prüfen und speichern
    result = build_result(parsed, text_result.text, settings.sum_tolerance, truncated, settings.own_names_list)
    save_result(receipt_id, result, llm_raw)
    log.info(
        "Bon %s: %s, %d Positionen, Summe %s, %s",
        receipt_id, result.merchant, len(result.items), result.total,
        "prüfen: " + "; ".join(result.review_notes) if result.needs_review else "ok",
    )
