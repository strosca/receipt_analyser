"""Hintergrund-Worker: arbeitet die Warteschlange Bon für Bon ab.

Es läuft immer nur EIN Bon gleichzeitig. Damit gibt es automatisch auch nur
eine KI-Anfrage gleichzeitig und der Server wird nicht überlastet.
"""
import asyncio
import logging
from typing import Awaitable, Callable

from .db import STATUS_ERROR, STATUS_PROCESSING, STATUS_QUEUED, get_db

log = logging.getLogger(__name__)

ProcessFn = Callable[[int], Awaitable[None]]


def claim_next() -> int | None:
    """Holt den ältesten wartenden Bon und setzt ihn auf 'verarbeitung'."""
    with get_db() as db:
        row = db.execute(
            """
            UPDATE receipts
               SET status = ?, status_detail = 'Gestartet', attempts = attempts + 1, updated_at = datetime('now')
             WHERE id = (SELECT id FROM receipts WHERE status = ? ORDER BY created_at, id LIMIT 1)
            RETURNING id
            """,
            (STATUS_PROCESSING, STATUS_QUEUED),
        ).fetchone()
        return row["id"] if row else None


def requeue_interrupted() -> int:
    """Nach einem Neustart: Bons, die mitten in der Verarbeitung waren, wieder einreihen."""
    with get_db() as db:
        cur = db.execute(
            "UPDATE receipts SET status = ?, status_detail = 'Nach Neustart erneut eingereiht' WHERE status = ?",
            (STATUS_QUEUED, STATUS_PROCESSING),
        )
        return cur.rowcount


def set_progress(receipt_id: int, detail: str) -> None:
    with get_db() as db:
        db.execute(
            "UPDATE receipts SET status_detail = ?, updated_at = datetime('now') WHERE id = ?", (detail, receipt_id)
        )


class Worker:
    def __init__(self, process: ProcessFn, idle_poll_seconds: float = 30):
        self._process = process
        self._wakeup = asyncio.Event()
        self._idle = idle_poll_seconds
        self._task: asyncio.Task | None = None
        self.current: int | None = None

    def notify(self) -> None:
        """Weckt den Worker sofort, z. B. nach einem Upload."""
        self._wakeup.set()

    def start(self) -> None:
        n = requeue_interrupted()
        if n:
            log.info("%d unterbrochene Bons wieder eingereiht", n)
        self._task = asyncio.create_task(self._run(), name="receipt-worker")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def run_once(self) -> bool:
        """Verarbeitet einen Bon, falls vorhanden. Gibt False zurück, wenn die Warteschlange leer ist."""
        receipt_id = await asyncio.to_thread(claim_next)
        if receipt_id is None:
            return False
        self.current = receipt_id
        log.info("Verarbeite Bon %s", receipt_id)
        try:
            await self._process(receipt_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 – jeder Fehler soll am Bon sichtbar werden
            log.exception("Bon %s: Verarbeitung fehlgeschlagen", receipt_id)
            with get_db() as db:
                db.execute(
                    "UPDATE receipts SET status = ?, status_detail = ?, updated_at = datetime('now') WHERE id = ?",
                    (STATUS_ERROR, f"{type(exc).__name__}: {exc}"[:2000], receipt_id),
                )
        finally:
            self.current = None
        return True

    async def _run(self) -> None:
        while True:
            self._wakeup.clear()
            try:
                had_work = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 – Worker darf nie sterben
                log.exception("Unerwarteter Fehler im Worker")
                had_work = False
            if not had_work:
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=self._idle)
                except asyncio.TimeoutError:
                    pass
