"""Texterkennung: PDF-Textebene direkt auslesen, sonst Bild aufbereiten + Tesseract.

Ablauf für Fotos von (oft zerknitterten) Thermopapier-Bons:
  1. Bild laden, EXIF-Drehung anwenden (iPhone), HEIC unterstützen
  2. Bon im Foto finden und perspektivisch entzerren (falls er sich vom Hintergrund abhebt)
  3. Graustufen, auf sinnvolle Größe skalieren
  4. Schatten/ungleichmäßige Beleuchtung entfernen (Hintergrund herausrechnen)
  5. Kontrast verstärken (CLAHE), leicht schärfen
  6. Schräglage korrigieren (Begradigen)
  7. Tesseract mit deu+eng, Seitenmodus "eine Textspalte"
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import get_settings

log = logging.getLogger(__name__)

try:  # HEIC-Fotos vom iPhone
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

SUBPROCESS_TIMEOUT = 300
PDF_MIN_TEXT_CHARS = 40  # weniger Buchstaben/Ziffern pro Seite => als Scan behandeln
TARGET_HEIGHT_RANGE = (1800, 4000)


@dataclass
class TextResult:
    text: str
    source: str  # 'pdf' oder 'ocr'
    pages: int


class OcrError(Exception):
    pass


# --------------------------------------------------------------------------- PDF


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=SUBPROCESS_TIMEOUT, check=True, **kw)
    except FileNotFoundError as exc:
        raise OcrError(f"Programm nicht gefunden: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode(errors="replace").strip()[:500]
        raise OcrError(f"{cmd[0]} fehlgeschlagen: {err}") from exc
    except subprocess.TimeoutExpired as exc:
        raise OcrError(f"{cmd[0]} hat zu lange gebraucht") from exc


def pdf_text_layer(path: Path) -> tuple[str, int]:
    """Liest die Textebene eines PDFs. Gibt (Text, Seitenzahl) zurück."""
    out = _run(["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"]).stdout.decode("utf-8", errors="replace")
    pages = out.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return out.replace("\f", "\n"), max(len(pages), 1)


def has_usable_text(text: str, pages: int) -> bool:
    alnum = sum(ch.isalnum() for ch in text)
    return alnum >= PDF_MIN_TEXT_CHARS * pages


def pdf_to_images(path: Path, out_dir: Path, dpi: int = 300) -> list[Path]:
    _run(["pdftoppm", "-r", str(dpi), "-gray", "-png", str(path), str(out_dir / "page")])
    return sorted(out_dir.glob("page*.png"))


# --------------------------------------------------------------------------- Bild


def load_image(path: Path) -> np.ndarray:
    """Lädt ein Bild als Graustufen-Array, mit korrekter Drehung laut EXIF."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        return np.array(img)


def _resize_to_range(gray: np.ndarray) -> np.ndarray:
    h = gray.shape[0]
    lo, hi = TARGET_HEIGHT_RANGE
    if h < lo:
        f = lo / h
        return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    if h > hi:
        f = hi / h
        return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return gray


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def find_and_warp_receipt(gray: np.ndarray) -> np.ndarray:
    """Sucht das helle Bon-Rechteck im Foto und entzerrt es. Ohne Treffer: Bild unverändert."""
    h, w = gray.shape
    scale = 800 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray.copy()
    scale = min(scale, 1.0)
    blur = cv2.GaussianBlur(small, (7, 7), 0)
    # Heller Bon auf dunklerem Grund: Otsu trennt beides, Morphologie schließt Textlücken
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    c = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(c) / float(small.shape[0] * small.shape[1])
    if not 0.15 < area_ratio < 0.97:
        return gray  # nichts gefunden oder Bon füllt das ganze Bild
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) != 4:
        # Kein sauberes Viereck (zerknittert): wenigstens auf das umschließende gedrehte Rechteck zuschneiden
        approx = cv2.boxPoints(cv2.minAreaRect(c))
    src = _order_corners(np.asarray(approx) / scale)
    tl, tr, br, bl = src
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 100 or height < 100:
        return gray
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def remove_shadows(gray: np.ndarray) -> np.ndarray:
    """Teilt das Bild durch seinen geschätzten Hintergrund -> gleichmäßig helles Papier."""
    k = max(15, (min(gray.shape) // 30) | 1)
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), k)
    norm = cv2.divide(gray, background, scale=255)
    return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)


def enhance(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(gray)
    blurred = cv2.GaussianBlur(out, (0, 0), 1.5)
    return cv2.addWeighted(out, 1.6, blurred, -0.6, 0)  # Unschärfemaske


def estimate_skew(gray: np.ndarray) -> float:
    """Schätzt die Schräglage der Textzeilen in Grad (positiv = gegen den Uhrzeigersinn)."""
    small_scale = 1000 / max(gray.shape)
    small = cv2.resize(gray, None, fx=small_scale, fy=small_scale) if small_scale < 1 else gray
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Buchstaben einer Zeile zu Balken verschmelzen
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    lines = cv2.HoughLinesP(bw, 1, np.pi / 720, threshold=80, minLineLength=small.shape[1] // 5, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 20:
            angles.append(a)
    if not angles:
        return 0.0
    return -float(np.median(angles))


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=255)


def preprocess(gray: np.ndarray) -> np.ndarray:
    gray = find_and_warp_receipt(gray)
    gray = _resize_to_range(gray)
    gray = remove_shadows(gray)
    gray = enhance(gray)
    angle = estimate_skew(gray)
    if 0.3 < abs(angle) < 15:
        gray = rotate(gray, angle)
    return gray


# --------------------------------------------------------------------------- Tesseract


def _osd_rotation(image_path: Path) -> int:
    """Fragt Tesseract, ob das Bild um 90/180/270° gedreht ist. 0 bei Unsicherheit."""
    try:
        out = _run(["tesseract", str(image_path), "stdout", "--psm", "0"]).stdout.decode(errors="replace")
    except OcrError:
        return 0
    m = re.search(r"Rotate:\s*(\d+)", out)
    conf = re.search(r"Orientation confidence:\s*([\d.]+)", out)
    if m and conf and float(conf.group(1)) > 2:
        return int(m.group(1))
    return 0


# Jeder Tesseract-Prozess nur 1 Thread – wir starten stattdessen mehrere Prozesse parallel
_TESS_ENV = {**os.environ, "OMP_THREAD_LIMIT": "1"}
MIN_STRIP_HEIGHT = 600


def tesseract(image_path: Path) -> str:
    lang = get_settings().ocr_lang
    cmd = [
        "tesseract", str(image_path), "stdout",
        "-l", lang, "--oem", "1", "--psm", "4", "--dpi", "300",
        "-c", "preserve_interword_spaces=1",
    ]
    return _run(cmd, env=_TESS_ENV).stdout.decode("utf-8", errors="replace")


def find_cut_rows(gray: np.ndarray, parts: int) -> list[int]:
    """Zeilen (y), an denen das Bild in `parts` Streifen geschnitten werden kann, ohne Text zu zerteilen.

    Gesucht wird jeweils nahe der idealen Schnittstelle eine (fast) weiße Zeile, möglichst
    mittig in einer Lücke zwischen zwei Textzeilen. Ohne Lücke wird dort nicht geschnitten.

    Echte Fotos haben Rauschen und Papierkanten, daher:
      - einzelne Pixel/Krümel werden vorher weggefiltert,
      - die äußeren 5 % links/rechts zählen nicht (Kante, Schatten),
      - eine Zeile mit höchstens 1 % Tinte gilt als Lücke.
    """
    h, w = gray.shape
    if parts <= 1 or h < 2 * MIN_STRIP_HEIGHT:
        return []
    parts = min(parts, h // MIN_STRIP_HEIGHT)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Rauschen weg
    margin = w // 20
    inner = ink[:, margin: w - margin]
    row_ink = inner.sum(axis=1)
    blank = row_ink <= max(2, inner.shape[1] // 100)
    cuts = []
    window = h // (parts * 3)
    for k in range(1, parts):
        target = k * h // parts
        lo, hi = max(target - window, 1), min(target + window, h - 1)
        best, best_len = None, 0
        y = lo
        while y < hi:  # längsten weißen Abschnitt im Fenster suchen
            if blank[y]:
                start = y
                while y < hi and blank[y]:
                    y += 1
                if y - start > best_len:
                    best, best_len = (start + y) // 2, y - start
            y += 1
        if best is not None and best_len >= 3 and (not cuts or best - cuts[-1] >= MIN_STRIP_HEIGHT // 2):
            cuts.append(best)
    return cuts


def tesseract_parallel(gray: np.ndarray, workdir: Path, name: str) -> str:
    """Bild in Streifen zwischen Textzeilen teilen und gleichzeitig erkennen lassen."""
    threads = max(1, get_settings().ocr_threads)
    cuts = find_cut_rows(gray, threads)
    bounds = [0, *cuts, gray.shape[0]]
    paths = []
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        path = workdir / f"{name}-part{i}.png"
        cv2.imwrite(str(path), gray[a:b])
        paths.append(path)
    if len(paths) == 1:
        if threads > 1:
            reason = "Bild zu klein" if gray.shape[0] < 2 * MIN_STRIP_HEIGHT else "keine Lücke zwischen Zeilen gefunden"
            log.info("OCR %s: nicht geteilt (%s, Höhe %d px), 1 Kern", name, reason, gray.shape[0])
        return tesseract(paths[0])
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        parts = list(pool.map(tesseract, paths))
    log.info("OCR %s: %d Streifen parallel", name, len(paths))
    return "\n".join(p.rstrip("\n") for p in parts)


def ocr_image(gray: np.ndarray, workdir: Path, name: str) -> str:
    # Querformat ist bei Bons verdächtig -> Drehung prüfen lassen
    if gray.shape[1] > gray.shape[0] * 1.2:
        probe = workdir / f"{name}-osd.png"
        cv2.imwrite(str(probe), gray)
        rot = _osd_rotation(probe)
        if rot:
            codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            gray = cv2.rotate(gray, codes[rot])
    processed = preprocess(gray)
    return tesseract_parallel(processed, workdir, name)


# --------------------------------------------------------------------------- Einstieg


def extract_text(path: Path, mime: str) -> TextResult:
    with tempfile.TemporaryDirectory(prefix="ocr-") as tmp:
        workdir = Path(tmp)
        if mime == "application/pdf":
            text, pages = pdf_text_layer(path)
            if has_usable_text(text, pages):
                return TextResult(text=text, source="pdf", pages=pages)
            log.info("%s: keine brauchbare Textebene, starte OCR", path.name)
            images = pdf_to_images(path, workdir)
            if not images:
                raise OcrError("PDF enthält keine Seiten")
            parts = [ocr_image(load_image(p), workdir, f"p{i}") for i, p in enumerate(images, 1)]
            return TextResult(text="\n".join(parts), source="ocr", pages=len(images))
        gray = load_image(path)
        return TextResult(text=ocr_image(gray, workdir, "img"), source="ocr", pages=1)
