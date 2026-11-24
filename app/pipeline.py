"""Verarbeitung eines Bons: Text gewinnen -> KI-Auswertung -> Prüfung -> speichern."""
import logging

log = logging.getLogger(__name__)


async def process_receipt(receipt_id: int) -> None:
    raise NotImplementedError("Texterkennung ist noch nicht eingebaut (folgt in Schritt 3).")
