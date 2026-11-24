import io

from app.db import get_db

from .helpers import PDF_BYTES, logged_in_client


def _upload(c):
    return c.post("/upload", files={"file": ("bon.pdf", io.BytesIO(PDF_BYTES), "application/pdf")}).json()["id"]


def _fake_processed(rid):
    with get_db() as db:
        db.execute(
            """UPDATE receipts SET status='pruefen', merchant='REWE', purchase_date='2026-11-03', total=3.00,
                   review_notes='Summe der Positionen 2,50 € ≠ Gesamtsumme 3,00 €', raw_text='BANANE 1,99'
               WHERE id=?""",
            (rid,),
        )
        db.executemany(
            "INSERT INTO items (receipt_id, position, name, price) VALUES (?, ?, ?, ?)",
            [(rid, 1, "BANANE", 1.99), (rid, 2, "LEERGUT", -0.25), (rid, 3, "MILCH", 0.76)],
        )


def test_list_and_filter():
    c = logged_in_client()
    rid = _upload(c)
    _fake_processed(rid)
    page = c.get("/bons").text
    assert "REWE" in page and "Prüfen" in page and "03.11.2026" in page
    assert "REWE" in c.get("/bons?status=pruefen").text
    assert "REWE" not in c.get("/bons?status=fertig").text


def test_detail_shows_items_notes_and_difference():
    c = logged_in_client()
    rid = _upload(c)
    _fake_processed(rid)
    page = c.get(f"/bons/{rid}").text
    assert "BANANE" in page and "-0,25 €" in page
    assert "Bitte prüfen" in page and "Gesamtsumme 3,00 €" in page
    assert "Abweichung zur Gesamtsumme" in page
    assert "BANANE 1,99" in page  # Rohtext
    assert c.get("/bons/9999").status_code == 404


def test_reprocess_and_delete():
    c = logged_in_client()
    rid = _upload(c)
    _fake_processed(rid)
    r = c.post(f"/bons/{rid}/neu", follow_redirects=False)
    assert r.status_code == 303
    with get_db() as db:
        assert db.execute("SELECT status FROM receipts WHERE id=?", (rid,)).fetchone()[0] == "warteschlange"
    c.post(f"/bons/{rid}/loeschen")
    with get_db() as db:
        assert db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
    assert c.get(f"/bons/{rid}/datei").status_code == 404
