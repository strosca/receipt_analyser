"""Verarbeitung eines Bons: Text gewinnen -> KI-Auswertung -> Prüfung -> speichern."""
import asyncio
import logging

from .db import STATUS_REVIEW, get_db
from .ocr import extract_text
from .storage import receipt_file_path
from .worker import set_progress

log = logging.getLogger(__name__)


async def process_receipt(receipt_id: int) -> None:
    with get_db() as db:
        r = db.execute("SELECT file_path, mime_type FROM receipts WHERE id = ?", (receipt_id,)).fetchone()

    set_progress(receipt_id, "Texterkennung läuft")
    result = await asyncio.to_thread(extract_text, receipt_file_path(r["file_path"]), r["mime_type"])
    log.info("Bon %s: %d Zeichen Text (%s, %d Seite/n)", receipt_id, len(result.text), result.source, result.pages)
    with get_db() as db:
        db.execute(
            "UPDATE receipts SET raw_text = ?, text_source = ?, updated_at = datetime('now') WHERE id = ?",
            (result.text, result.source, receipt_id),
        )

    # KI-Auswertung folgt in Schritt 4
    with get_db() as db:
        db.execute(
            "UPDATE receipts SET status = ?, status_detail = 'Text erkannt, KI-Auswertung noch nicht eingebaut' WHERE id = ?",
            (STATUS_REVIEW, receipt_id),
        )
