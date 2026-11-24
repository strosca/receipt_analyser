"""SQLite-Datenbank: Verbindung, Schema und Migrationen.

Wir nutzen die Standardbibliothek sqlite3 ohne ORM. Das Schema wird über
PRAGMA user_version versioniert: Jede Migration in MIGRATIONS läuft genau einmal.
"""
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from .config import get_settings

STATUS_QUEUED = "warteschlange"
STATUS_PROCESSING = "verarbeitung"
STATUS_REVIEW = "pruefen"
STATUS_DONE = "fertig"
STATUS_ERROR = "fehler"

STATUS_LABELS = {
    STATUS_QUEUED: "In Warteschlange",
    STATUS_PROCESSING: "Wird verarbeitet",
    STATUS_REVIEW: "Prüfen",
    STATUS_DONE: "Fertig",
    STATUS_ERROR: "Fehler",
}

MIGRATIONS: list[str] = [
    # 1: Bons und Positionen
    """
    CREATE TABLE receipts (
        id              INTEGER PRIMARY KEY,
        file_path       TEXT NOT NULL,          -- relativ zu data/uploads
        original_name   TEXT,
        mime_type       TEXT NOT NULL,
        file_hash       TEXT NOT NULL UNIQUE,   -- SHA-256, verhindert Duplikate
        status          TEXT NOT NULL DEFAULT 'warteschlange',
        status_detail   TEXT,                   -- aktueller Schritt bzw. Fehlermeldung
        review_notes    TEXT,                   -- Gründe für "prüfen", eine pro Zeile
        attempts        INTEGER NOT NULL DEFAULT 0,
        raw_text        TEXT,
        text_source     TEXT,                   -- 'pdf' oder 'ocr'
        llm_raw         TEXT,                   -- rohe KI-Antwort (zur Fehlersuche)
        merchant        TEXT,
        branch          TEXT,
        purchase_date   TEXT,                   -- ISO: JJJJ-MM-TT
        total           REAL,
        payment_method  TEXT,
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
        processed_at    TEXT
    );
    CREATE INDEX idx_receipts_status ON receipts(status, created_at);
    CREATE INDEX idx_receipts_date ON receipts(purchase_date);

    CREATE TABLE items (
        id          INTEGER PRIMARY KEY,
        receipt_id  INTEGER NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
        position    INTEGER NOT NULL,
        name        TEXT NOT NULL,
        price       REAL NOT NULL               -- Gesamtpreis der Zeile, negativ bei Rabatt/Leergut
    );
    CREATE INDEX idx_items_receipt ON items(receipt_id);
    """,
]


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_settings().db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    """Öffnet eine Verbindung, committet am Ende bzw. rollt bei Fehlern zurück."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    conn = connect()
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for i, sql in enumerate(MIGRATIONS[version:], start=version + 1):
            conn.executescript(f"BEGIN; {sql}; PRAGMA user_version = {i}; COMMIT;")
    finally:
        conn.close()
