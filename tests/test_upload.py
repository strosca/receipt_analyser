import asyncio
import io

from app.db import get_db, init_db
from app.storage import sniff_mime
from app.worker import Worker, claim_next, requeue_interrupted

from .helpers import JPEG_BYTES, PDF_BYTES, logged_in_client


def test_sniff_mime():
    assert sniff_mime(PDF_BYTES) == "application/pdf"
    assert sniff_mime(JPEG_BYTES) == "image/jpeg"
    assert sniff_mime(b"\x00\x00\x00\x18ftypheic....") == "image/heic"
    assert sniff_mime(b"hello world") is None


def test_upload_and_duplicate():
    c = logged_in_client()
    r = c.post("/upload", files={"file": ("bon.pdf", io.BytesIO(PDF_BYTES), "application/pdf")})
    assert r.status_code == 200 and r.json()["duplicate"] is False
    rid = r.json()["id"]
    r2 = c.post("/upload", files={"file": ("nochmal.pdf", io.BytesIO(PDF_BYTES), "application/pdf")})
    assert r2.json() == {"ok": True, "id": rid, "duplicate": True}
    assert c.get(f"/bons/{rid}/datei").content == PDF_BYTES
    assert "bon.pdf" in c.get("/partials/recent").text


def test_upload_rejects_unknown_type():
    c = logged_in_client()
    r = c.post("/upload", files={"file": ("x.txt", io.BytesIO(b"nur text"), "text/plain")})
    assert r.status_code == 400
    assert "nicht unterstützt" in r.json()["error"]


def _insert(status="warteschlange", h="a"):
    with get_db() as db:
        return db.execute(
            "INSERT INTO receipts (file_path, mime_type, file_hash, status) VALUES ('x', 'application/pdf', ?, ?)",
            (h, status),
        ).lastrowid


def test_queue_order_and_requeue():
    init_db()
    a, b = _insert(h="a"), _insert(h="b")
    assert claim_next() == a
    assert claim_next() == b
    assert claim_next() is None
    assert requeue_interrupted() == 2


def test_worker_marks_errors():
    init_db()
    rid = _insert()

    async def boom(_):
        raise RuntimeError("kaputt")

    assert asyncio.run(Worker(boom).run_once()) is True
    with get_db() as db:
        row = db.execute("SELECT status, status_detail FROM receipts WHERE id = ?", (rid,)).fetchone()
    assert row["status"] == "fehler"
    assert "kaputt" in row["status_detail"]
