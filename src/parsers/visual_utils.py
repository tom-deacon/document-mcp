"""
Visual element utilities: OCR, caption detection, and chunk construction.

This module is the single place that decides:
  - whether OCR is available (checked once at import time)
  - how to run OCR on raw image bytes
  - how to recognise figure/chart/diagram captions in text
  - how to build a standardised visual chunk dict

OCR requires  pytesseract  AND the Tesseract binary to be installed.
If either is missing the rest of the pipeline still works — visual chunks
are created but their ocr_text field will be empty.

Install guide (Windows):
  1. Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
     and run the installer (adds tesseract.exe to PATH).
  2. pip install pytesseract Pillow   (or: uv add pytesseract Pillow)
"""

import io
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# One-time availability checks
# ---------------------------------------------------------------------------

_PIL_AVAILABLE = False
_OCR_AVAILABLE = False

try:
    from PIL import Image as _PILImage   # noqa: F401
    _PIL_AVAILABLE = True
except ImportError:
    pass

if _PIL_AVAILABLE:
    try:
        import pytesseract as _pytesseract
        _pytesseract.get_tesseract_version()   # raises if binary missing
        _OCR_AVAILABLE = True
        logger.info("[VISUAL] Tesseract OCR ready (pytesseract)")
    except ImportError:
        logger.info(
            "[VISUAL] pytesseract not installed — OCR disabled. "
            "Run: uv add pytesseract Pillow  (also install Tesseract binary)"
        )
    except Exception as exc:
        logger.warning(
            "[VISUAL] pytesseract installed but Tesseract binary not found or "
            "not on PATH: %s. "
            "Download from https://github.com/UB-Mannheim/tesseract/wiki", exc
        )
else:
    logger.debug("[VISUAL] Pillow not available — OCR and image resize disabled")


def ocr_is_available() -> bool:
    """Return True if both pytesseract and the Tesseract binary are usable."""
    return _OCR_AVAILABLE


# ---------------------------------------------------------------------------
# Configuration dataclass (passed into parsers)
# ---------------------------------------------------------------------------

@dataclass
class VisualConfig:
    """Settings for visual element processing."""
    enable_ocr: bool = True
    ocr_language: str = "eng"
    # placeholder — set True + vision_model when you want AI image descriptions
    enable_vision_summary: bool = False
    vision_model: str = ""              # e.g. "llava:7b" via Ollama
    # OCR is skipped for images below this byte size (avoids wasting time on
    # logos, bullets, tiny decorative images).
    min_ocr_image_bytes: int = 5_000
    # Safety cap: OCR at most this many images per PDF page.
    max_images_per_page: int = 8


# ---------------------------------------------------------------------------
# Caption pattern detection
# ---------------------------------------------------------------------------

# Matches lines like:  "Figure 3:", "Fig. 2 —", "Chart 1:", "Diagram A:"
_CAPTION_RE = re.compile(
    r"^(?:fig(?:ure)?|chart|diagram|image|plate|exhibit|photo|illustration)\s*"
    r"[\dA-Za-z.]+\s*[:\-–]",
    re.IGNORECASE,
)


def is_caption_text(text: str) -> bool:
    """Return True if *text* looks like a figure/chart/diagram caption label."""
    return bool(_CAPTION_RE.match(text.strip()))


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def run_ocr(image_bytes: bytes, lang: str = "eng") -> str:
    """
    Run Tesseract OCR on raw image bytes.

    Returns the extracted text stripped of leading/trailing whitespace,
    or "" if OCR is unavailable, the image is empty, or an error occurs.
    Errors are logged at DEBUG level (expected when diagram contains no text).
    """
    if not _OCR_AVAILABLE or not _PIL_AVAILABLE or not image_bytes:
        return ""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        # Tesseract works best on RGB or greyscale
        if img.mode not in ("RGB", "L", "RGBA"):
            img = img.convert("RGB")
        return pytesseract.image_to_string(img, lang=lang).strip()
    except Exception as exc:
        logger.debug("[OCR] Failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Visual chunk builder
# ---------------------------------------------------------------------------

def build_visual_chunk(
    document_title: str,
    section_path: str,
    page_or_sheet: Optional[str],
    alt_text: str,
    caption: str,
    ocr_text: str,
    figure_label: str = "",     # e.g. "Figure 3" if known
    element_type: str = "figure",
) -> Dict[str, Any]:
    """
    Construct a standardised visual chunk dict.

    ``text``           — human-readable, stored as chunk_text in the DB.
    ``embedding_text`` — enriched with document + section context for the
                         embedding model.

    The extra keys (alt_text, caption, ocr_text) are *not* written to the
    LanceDB schema — they are present in the dict only for logging / the
    test_ingestion.py inspection script.
    """
    # --- plain-text representation (what gets stored in the DB) ---
    parts: List[str] = []
    if figure_label:
        parts.append(figure_label)
    if alt_text:
        parts.append(f"Description: {alt_text}")
    if caption:
        parts.append(f"Caption: {caption}")
    if ocr_text:
        parts.append(f"Text in image: {ocr_text}")
    if not parts:
        parts.append(f"[Visual element — {document_title or 'unknown document'}]")
    text = "\n".join(parts)

    # --- embedding text (richer, for the vector model) ---
    embed_parts: List[str] = []
    if document_title:
        embed_parts.append(f"Document: {document_title}")
    if section_path:
        embed_parts.append(f"Section: {section_path}")
    embed_parts.append(f"Type: {element_type}")
    if alt_text:
        embed_parts.append(f"Alt: {alt_text}")
    if caption:
        embed_parts.append(f"Caption: {caption}")
    if ocr_text:
        embed_parts.append(f"OCR: {ocr_text[:600]}")
    embedding_text = " | ".join(embed_parts)

    return {
        "element_type": element_type,
        "text": text,
        "section_path": section_path,
        "page_or_sheet": page_or_sheet or "",
        "html_or_markdown": "",
        "embedding_text": embedding_text,
        "token_count": 0,        # chunker will recalculate if needed
        "char_count": len(text),
        "start_pos": -1,
        "end_pos": -1,
        # Inspection fields — ignored by the indexer
        "alt_text": alt_text,
        "caption": caption,
        "ocr_text": ocr_text,
    }


def refresh_figure_text(figure_elem: Dict[str, Any], document_title: str) -> None:
    """Re-build text + embedding_text on a figure element after caption is known."""
    updated = build_visual_chunk(
        document_title=document_title,
        section_path=figure_elem.get("section_path", ""),
        page_or_sheet=figure_elem.get("page_or_sheet"),
        alt_text=figure_elem.get("alt_text", ""),
        caption=figure_elem.get("caption", ""),
        ocr_text=figure_elem.get("ocr_text", ""),
        figure_label=figure_elem.get("figure_label", ""),
        element_type=figure_elem.get("element_type", "figure"),
    )
    figure_elem["text"] = updated["text"]
    figure_elem["embedding_text"] = updated["embedding_text"]
    figure_elem["char_count"] = updated["char_count"]
