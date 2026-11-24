"""Seiten und Aktionen der Web-Oberfläche (alle nur nach Anmeldung)."""
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

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
