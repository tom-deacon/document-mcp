"""
Vision enhancement pass for PDF pages using the Anthropic API.

For each page with low text density (fewer than *word_threshold* words), the
page is rasterised to a PNG and sent to a Claude vision model, which produces
a structured prose description of the visible slide content.

Public API
----------
    enhance_pdf(file_path, word_threshold=50, api_key=None, dpi=150)
        -> list[dict]

Returns a list of element dicts (one per visually-rich page) with
    element_type = "vision_description"
    extracted_by = "vision"

These can be merged into the element list produced by the PDF parser before
the structure-aware chunker runs, or appended after regular parsing.
"""

import base64
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF — already in project dependencies

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL = "claude-sonnet-4-6"
_DPI = 150
_MAX_TOKENS = 1024

_VISION_SYSTEM_PROMPT = (
    "You are analysing a slide from a PDF document that originated from a "
    "PowerPoint presentation. Describe all content visible on this slide in "
    "detail, including: any frameworks, process flows, or diagrams and the "
    "relationships between their elements; any text labels, headings, or "
    "callouts; any charts or tables and their data; any icons with associated "
    "labels. Write your description as structured prose that would allow "
    "someone to fully understand the slide content without seeing the image. "
    "Do not include preamble."
)


# ---------------------------------------------------------------------------
# Page-level helpers
# ---------------------------------------------------------------------------

def _page_word_count(page: fitz.Page) -> int:
    """Return the number of whitespace-delimited words on *page*."""
    text = page.get_text().strip()
    return len(text.split()) if text else 0


def _rasterise_page(page: fitz.Page, dpi: int = _DPI) -> bytes:
    """Render *page* to a PNG byte string at *dpi* dots-per-inch."""
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return pixmap.tobytes("png")


def _call_vision_api(image_bytes: bytes, api_key: str) -> str:
    """Send *image_bytes* (PNG) to the Anthropic vision API and return the description."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")

    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_VISION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe this slide.",
                    },
                ],
            }
        ],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enhance_pdf(
    file_path: Path,
    word_threshold: int = 50,
    api_key: Optional[str] = None,
    dpi: int = _DPI,
) -> List[Dict[str, Any]]:
    """
    Generate vision-description elements for visually-rich pages in *file_path*.

    Parameters
    ----------
    file_path       : path to the PDF to analyse
    word_threshold  : pages with fewer words than this are sent to the vision API
    api_key         : Anthropic API key; falls back to the ANTHROPIC_API_KEY env var
    dpi             : rasterisation resolution (150 DPI is sufficient for legibility)

    Returns
    -------
    List of element dicts — one per visually-rich page — ready to be merged
    into the PDF parser's element list before chunking.  Each dict has:
        element_type  = "vision_description"
        text          = "[Vision description of slide N]: <prose>"
        section_path  = "Page N"
        page_or_sheet = "N"
        extracted_by  = "vision"
    """
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not resolved_key:
        logger.warning(
            "[VISION] ANTHROPIC_API_KEY not set — vision enhancement skipped for %s",
            file_path.name,
        )
        return []

    try:
        import anthropic  # noqa: F401  — verify the package is installed
    except ImportError:
        logger.warning(
            "[VISION] 'anthropic' package not installed — vision enhancement skipped. "
            "Install with: uv add anthropic"
        )
        return []

    vision_elements: List[Dict[str, Any]] = []

    try:
        with fitz.open(str(file_path)) as pdf:
            total = len(pdf)
            for page_idx in range(total):
                page_num = page_idx + 1  # 1-indexed to match Docling convention
                page = pdf[page_idx]

                word_count = _page_word_count(page)
                if word_count >= word_threshold:
                    logger.debug(
                        "[VISION] Page %d/%d: %d words — text-dense, skipping",
                        page_num, total, word_count,
                    )
                    continue

                print(
                    f"[VISION] Page {page_num}/{total}: {word_count} words "
                    f"(< {word_threshold} threshold) — calling Claude vision …",
                    file=sys.stderr, flush=True,
                )

                try:
                    image_bytes = _rasterise_page(page, dpi=dpi)
                    description = _call_vision_api(image_bytes, resolved_key)
                except Exception as exc:
                    logger.warning(
                        "[VISION] Page %d: API call failed — %s: %s",
                        page_num, type(exc).__name__, exc,
                    )
                    continue

                chunk_text = f"[Vision description of slide {page_num}]: {description}"
                vision_elements.append({
                    "element_type": "vision_description",
                    "text": chunk_text,
                    "section_path": f"Page {page_num}",
                    "page_or_sheet": str(page_num),
                    "html_or_markdown": "",
                    "extracted_by": "vision",
                })

                print(
                    f"[VISION] Page {page_num}: description generated "
                    f"({len(description):,} chars)",
                    file=sys.stderr, flush=True,
                )

    except Exception as exc:
        logger.error(
            "[VISION] Failed to process %s — %s: %s",
            file_path.name, type(exc).__name__, exc,
        )

    logger.info(
        "[VISION] %s: %d vision-description element(s) generated",
        file_path.name, len(vision_elements),
    )
    return vision_elements
