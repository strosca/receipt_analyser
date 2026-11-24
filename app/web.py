"""Seiten und Aktionen der Web-Oberfläche (alle nur nach Anmeldung)."""
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from .auth import require_login
from .db import STATUS_LABELS, get_db
from .storage import UploadError, receipt_file_path, save_upload
from .templating import templates

router = APIRouter(dependencies=[Depends(require_login)])


def _worker(request: Request):
    return request.app.state.worker


@router.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse(request, "upload.html", {"active": "upload"})


@router.post("/upload")
async def upload(request: Request, file: UploadFile):
    try:
        saved = save_upload(file.file, file.filename)
    except UploadError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    if not saved.duplicate:
        _worker(request).notify()
    return {"ok": True, "id": saved.receipt_id, "duplicate": saved.duplicate}


@router.get("/partials/recent", response_class=HTMLResponse)
async def recent_partial(request: Request):
    with get_db() as db:
        rows = db.execute(
            """SELECT id, original_name, status, status_detail, merchant, purchase_date, total, created_at
                 FROM receipts ORDER BY id DESC LIMIT 10"""
        ).fetchall()
    return templates.TemplateResponse(
        request, "partials/receipt_rows.html", {"receipts": rows, "labels": STATUS_LABELS}
    )


@router.get("/bons/{receipt_id}/datei")
async def receipt_file(receipt_id: int):
    with get_db() as db:
        row = db.execute("SELECT file_path, mime_type FROM receipts WHERE id = ?", (receipt_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    path = receipt_file_path(row["file_path"])
    if not path.exists():
        raise HTTPException(404, "Datei fehlt")
    return FileResponse(path, media_type=row["mime_type"])


# --------------------------------------------------------------------------- Bon-Liste

def _list_receipts(status: str | None):
    sql = """SELECT r.id, r.original_name, r.status, r.status_detail, r.merchant, r.purchase_date, r.total,
                    r.created_at, (SELECT COUNT(*) FROM items i WHERE i.receipt_id = r.id) AS item_count
               FROM receipts r"""
    params: tuple = ()
    if status:
        sql += " WHERE r.status = ?"
        params = (status,)
    sql += " ORDER BY r.id DESC LIMIT 300"
    with get_db() as db:
        rows = db.execute(sql, params).fetchall()
        counts = dict(db.execute("SELECT status, COUNT(*) FROM receipts GROUP BY status").fetchall())
    return rows, counts


@router.get("/bons", response_class=HTMLResponse)
async def receipts_page(request: Request, status: str | None = None):
    status = status if status in STATUS_LABELS else None
    rows, counts = _list_receipts(status)
    return templates.TemplateResponse(
        request,
        "receipts.html",
        {"active": "bons", "receipts": rows, "counts": counts, "labels": STATUS_LABELS, "status": status},
    )


@router.get("/partials/bons", response_class=HTMLResponse)
async def receipts_partial(request: Request, status: str | None = None):
    status = status if status in STATUS_LABELS else None
    rows, _ = _list_receipts(status)
    return templates.TemplateResponse(
        request, "partials/receipt_rows.html", {"receipts": rows, "labels": STATUS_LABELS}
    )


# --------------------------------------------------------------------------- Bon-Detail

def _load_receipt(receipt_id: int):
    with get_db() as db:
        r = db.execute("SELECT * FROM receipts WHERE id = ?", (receipt_id,)).fetchone()
        if not r:
            raise HTTPException(404, "Bon nicht gefunden")
        items = db.execute(
            "SELECT * FROM items WHERE receipt_id = ? ORDER BY position", (receipt_id,)
        ).fetchall()
    return r, items


@router.get("/bons/{receipt_id}", response_class=HTMLResponse)
async def receipt_detail(request: Request, receipt_id: int):
    r, items = _load_receipt(receipt_id)
    items_sum = round(sum(i["price"] for i in items), 2)
    return templates.TemplateResponse(
        request,
        "receipt_detail.html",
        {
            "active": "bons",
            "r": r,
            "items": items,
            "items_sum": items_sum,
            "labels": STATUS_LABELS,
            "review_notes": (r["review_notes"] or "").splitlines(),
            "busy": r["status"] in ("warteschlange", "verarbeitung"),
        },
    )


@router.post("/bons/{receipt_id}/neu")
async def reprocess(request: Request, receipt_id: int):
    with get_db() as db:
        cur = db.execute(
            """UPDATE receipts SET status = 'warteschlange', status_detail = 'Erneut eingereiht',
                   updated_at = datetime('now')
               WHERE id = ? AND status != 'verarbeitung'""",
            (receipt_id,),
        )
    if cur.rowcount:
        _worker(request).notify()
    return RedirectResponse(f"/bons/{receipt_id}", status_code=303)


@router.post("/bons/{receipt_id}/loeschen")
async def delete_receipt(receipt_id: int):
    with get_db() as db:
        row = db.execute("SELECT file_path, status FROM receipts WHERE id = ?", (receipt_id,)).fetchone()
        if not row:
            raise HTTPException(404)
        if row["status"] == "verarbeitung":
            raise HTTPException(409, "Bon wird gerade verarbeitet")
        db.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))
    receipt_file_path(row["file_path"]).unlink(missing_ok=True)
    return RedirectResponse("/bons", status_code=303)
