"""
ocr_engine.py — Computer Vision module for extracting text from exam images/PDFs.

Supports:
  • EasyOCR   (default, no external binary needed)
  • Tesseract (requires tesseract-ocr installed on system)
  • PDF pages are converted to images, then OCR-ed.

Returns structured text: list of (question_number, answer_text) when the format
is detectable, or raw text otherwise.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)

# ── Lazy-loaded globals (heavy imports) ──────────────────────────────────────
_easyocr_reader = None
_tesseract_available = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Low-level OCR backends
# ═══════════════════════════════════════════════════════════════════════════════

def _get_easyocr_reader():
    """Lazily initialise EasyOCR reader (downloads model on first call)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(
            config.OCR_LANGUAGES,
            gpu=True,          # will fall back to CPU automatically
            verbose=False,
        )
        logger.info("EasyOCR reader initialised (langs=%s)", config.OCR_LANGUAGES)
    return _easyocr_reader


def ocr_easyocr(image: np.ndarray) -> str:
    """Run EasyOCR on a numpy image array and return joined text."""
    reader = _get_easyocr_reader()
    results = reader.readtext(image, detail=0, paragraph=True)
    return "\n".join(results)


def ocr_tesseract(image: np.ndarray) -> str:
    """Run Tesseract OCR on a numpy image array."""
    import pytesseract
    pil_img = Image.fromarray(image)
    lang_str = "+".join(config.OCR_LANGUAGES)
    text = pytesseract.image_to_string(pil_img, lang=lang_str)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  Image / PDF loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_image(path: str | Path) -> np.ndarray:
    """Load an image file and return a numpy RGB array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def extract_pdf_text_native(path: str | Path) -> Optional[str]:
    """
    Try to extract embedded text directly from a digital/typed PDF.
    Returns the text if enough content is found, else None (= needs OCR).
    """
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    texts = []
    for page in doc:
        texts.append(page.get_text())
    doc.close()
    full = "\n\n".join(texts).strip()
    # If the PDF has meaningful embedded text (>30 chars), use it directly
    if len(full) > 30:
        logger.info("PDF has embedded text (%d chars) — skipping OCR.", len(full))
        return full
    return None


def load_pdf_pages(path: str | Path, dpi: int = 150) -> List[np.ndarray]:
    """Convert each PDF page to a numpy RGB image using PyMuPDF (no Poppler needed).
    Default dpi=150 for speed; increase to 300 for handwritten exams."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    images = []
    zoom = dpi / 72  # 72 is the default PDF DPI
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        images.append(img.copy())  # copy to own the memory
    doc.close()
    return images


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Optional preprocessing to improve OCR quality:
      - Convert to grayscale
      - Apply adaptive threshold
      - Slight denoise
    """
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Adaptive threshold for binarisation
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )
    # Convert back to 3-channel for OCR engines that expect RGB
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


# ═══════════════════════════════════════════════════════════════════════════════
#  Answer segmentation
# ═══════════════════════════════════════════════════════════════════════════════

def segment_answers(raw_text: str) -> List[Tuple[str, str]]:
    """
    Try to split OCR text into (question_id, answer_text) pairs.

    Looks for patterns like:
      Q1: …  |  Question 1: …  |  1) …  |  1. …  |  **1** …
    If no pattern is found, returns the full text as a single answer.
    """
    # Pattern: number (optionally preceded by Q/Question) followed by separator
    pattern = r'(?:^|\n)\s*(?:Q(?:uestion)?\s*)?(\d+)\s*[).::\-]\s*'
    splits = re.split(pattern, raw_text, flags=re.IGNORECASE)

    if len(splits) < 3:
        # Could not segment — return as single blob
        return [("1", raw_text.strip())]

    pairs: List[Tuple[str, str]] = []
    # splits = [preamble, num1, text1, num2, text2, ...]
    for i in range(1, len(splits) - 1, 2):
        q_id = splits[i].strip()
        answer = splits[i + 1].strip()
        if answer:
            pairs.append((q_id, answer))

    return pairs if pairs else [("1", raw_text.strip())]


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(file_path: str | Path, preprocess: bool = True) -> str:
    """
    Extract raw text from an image or PDF file.

    Parameters
    ----------
    file_path : path to .jpg, .png, .bmp, .tiff, or .pdf
    preprocess : whether to apply image preprocessing before OCR

    Returns
    -------
    Full OCR-ed text as a string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    ocr_fn = ocr_easyocr if config.OCR_ENGINE == "easyocr" else ocr_tesseract

    if ext == ".pdf":
        # First try to extract embedded text (instant for digital/typed PDFs)
        native_text = extract_pdf_text_native(path)
        if native_text:
            return native_text
        # Fallback: render pages to images and run OCR (for scanned/handwritten)
        logger.info("No embedded text — running OCR on PDF pages (dpi=150)...")
        pages = load_pdf_pages(path)
        texts = []
        for i, page_img in enumerate(pages):
            if preprocess:
                page_img = preprocess_image(page_img)
            text = ocr_fn(page_img)
            texts.append(text)
            logger.info("OCR page %d/%d: %d chars", i + 1, len(pages), len(text))
        return "\n\n".join(texts)
    else:
        image = load_image(path)
        if preprocess:
            image = preprocess_image(image)
        return ocr_fn(image)


def extract_answers(file_path: str | Path, preprocess: bool = True) -> List[Tuple[str, str]]:
    """
    Extract text from an exam image/PDF, then segment into individual answers.

    Returns
    -------
    List of (question_id, answer_text) tuples.
    """
    raw = extract_text(file_path, preprocess=preprocess)
    return segment_answers(raw)


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python ocr_engine.py <image_or_pdf_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"\n{'='*60}")
    print(f"  OCR Engine: {config.OCR_ENGINE}")
    print(f"  File: {file_path}")
    print(f"{'='*60}\n")

    answers = extract_answers(file_path)
    for qid, ans in answers:
        print(f"  Q{qid}: {ans[:200]}{'...' if len(ans) > 200 else ''}")
        print()
