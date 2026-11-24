import cv2
import numpy as np
import pytest

from app import ocr


def fake_receipt(height=2400, width=900):
    """Weißes Papier mit schwarzen Textzeilen."""
    img = np.full((height, width), 255, np.uint8)
    for i, y in enumerate(range(100, height - 100, 60)):
        cv2.putText(img, f"ARTIKEL NR {i:03d}      {i},99 A", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2)
    return img


@pytest.mark.parametrize("angle", [-6.0, -2.0, 3.0, 7.0])
def test_skew_is_corrected(angle):
    rotated = ocr.rotate(fake_receipt(), angle)
    est = ocr.estimate_skew(rotated)
    fixed = ocr.rotate(rotated, est)
    assert abs(ocr.estimate_skew(fixed)) < 0.8, (angle, est)


def test_receipt_is_found_on_dark_background():
    bg = np.full((3000, 2400), 60, np.uint8)
    rec = fake_receipt()
    bg[300:300 + rec.shape[0], 700:700 + rec.shape[1]] = rec
    warped = ocr.find_and_warp_receipt(bg)
    h, w = warped.shape
    assert abs(h - rec.shape[0]) < 60 and abs(w - rec.shape[1]) < 60
    assert warped.mean() > 200  # fast nur Papier, kein dunkler Hintergrund


def test_preprocess_runs_and_returns_gray():
    out = ocr.preprocess(ocr.rotate(fake_receipt(), 4))
    assert out.ndim == 2 and out.dtype == np.uint8


def test_shadow_removal_evens_out_background():
    img = fake_receipt().astype(np.float32)
    gradient = np.linspace(0.45, 1.0, img.shape[1], dtype=np.float32)[None, :]
    shaded = (img * gradient).astype(np.uint8)
    out = ocr.remove_shadows(shaded)
    left, right = out[:, :50].mean(), out[:, -50:].mean()
    assert abs(left - right) < 25


def test_pdf_text_threshold():
    assert ocr.has_usable_text("REWE Markt GmbH SUMME EUR 12,34 " * 3, 1)
    assert not ocr.has_usable_text("  \n \n ", 1)


# ---------------------------------------------------------------- parallele Texterkennung

def _ink_rows(img):
    return (img < 128).sum(axis=1)


@pytest.mark.parametrize("parts", [2, 3, 4])
def test_cuts_only_between_text_lines(parts):
    img = ocr.preprocess(fake_receipt(height=3000))
    cuts = ocr.find_cut_rows(img, parts)
    assert len(cuts) == parts - 1
    ink = _ink_rows(img)
    for y in cuts:
        # Schnittzeile und ihre Nachbarn sind weiß -> keine Textzeile zerteilt
        assert ink[y - 2: y + 3].sum() == 0, y


def test_no_cut_without_gap():
    # Komplett schwarz gestreiftes Bild ohne weiße Zeile -> nicht schneiden
    img = np.full((3000, 800), 255, np.uint8)
    img[::2, :] = 0
    assert ocr.find_cut_rows(img, 3) == []


def test_small_image_is_not_split():
    assert ocr.find_cut_rows(fake_receipt(height=900), 3) == []


def test_parallel_ocr_keeps_order(tmp_path, monkeypatch):
    import threading
    import time

    active, peak = 0, 0
    lock = threading.Lock()

    def fake_tesseract(path):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.2)
        with lock:
            active -= 1
        return f"TEIL {path.stem[-1]}\n"

    monkeypatch.setattr(ocr, "tesseract", fake_tesseract)
    text = ocr.tesseract_parallel(ocr.preprocess(fake_receipt(height=3000)), tmp_path, "img")
    assert text == "TEIL 0\nTEIL 1\nTEIL 2"
    assert peak == 3  # wirklich gleichzeitig
