"""Speichern hochgeladener Dateien und Anlegen der Bons in der Datenbank."""
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from .config import get_settings
from .db import STATUS_QUEUED, get_db

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

EXTENSIONS = {
    "application/pdf": "pdf",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/heic": "heic",
    "image/webp": "webp",
    "image/tiff": "tif",
}


class UploadError(Exception):
    pass


def sniff_mime(head: bytes) -> str | None:
    """Erkennt den Dateityp am Inhalt (nicht an der Endung, die oft fehlt oder falsch ist)."""
    if head.startswith(b"%PDF"):
        return "application/pdf"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"mif1", b"msf1", b"heim", b"heis"):
        return "image/heic"
    return None


@dataclass
class SavedUpload:
    receipt_id: int
    duplicate: bool


def save_upload(stream: BinaryIO, original_name: str | None) -> SavedUpload:
    settings = get_settings()
    settings.ensure_dirs()
    sha = hashlib.sha256()
    size = 0
    head = b""
    fd, tmp_name = tempfile.mkstemp(dir=settings.upload_dir, prefix=".upload-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := stream.read(1024 * 1024):
                if len(head) < 32:
                    head += chunk[: 32 - len(head)]
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise UploadError("Datei ist größer als 50 MB.")
                sha.update(chunk)
                out.write(chunk)
        if size == 0:
            raise UploadError("Datei ist leer.")
        mime = sniff_mime(head)
        if mime is None:
            raise UploadError("Dateityp nicht unterstützt. Erlaubt sind PDF, JPG, PNG, HEIC, WebP.")
        digest = sha.hexdigest()

        with get_db() as db:
            row = db.execute("SELECT id FROM receipts WHERE file_hash = ?", (digest,)).fetchone()
            if row:
                return SavedUpload(receipt_id=row["id"], duplicate=True)

            rel = Path(datetime.now().strftime("%Y-%m")) / f"{digest[:20]}.{EXTENSIONS[mime]}"
            target = settings.upload_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, target)
            cur = db.execute(
                "INSERT INTO receipts (file_path, original_name, mime_type, file_hash, status) VALUES (?, ?, ?, ?, ?)",
                (rel.as_posix(), original_name, mime, digest, STATUS_QUEUED),
            )
            log.info("Bon %s hochgeladen (%s, %d Bytes)", cur.lastrowid, mime, size)
            return SavedUpload(receipt_id=cur.lastrowid, duplicate=False)
    finally:
        tmp.unlink(missing_ok=True)


def receipt_file_path(file_path: str) -> Path:
    return get_settings().upload_dir / file_path
