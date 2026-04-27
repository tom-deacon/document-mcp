"""Enhanced PDF parser — extracts block-level elements with page metadata."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Pages where the text-to-image ratio is below this threshold are flagged as
# image-heavy (likely diagram/scan pages requiring OCR for full content).
_IMAGE_PAGE_CHAR_THRESHOLD = 80


def _block_element_type(block: tuple) -> str:
    """Classify a PyMuPDF block tuple as 'figure' or 'paragraph'."""
    # block = (x0, y0, x1, y1, text, block_no, block_type)
    # block_type: 0 = text, 1 = image
    block_type = block[6] if len(block) > 6 else 0
    return "figure" if block_type == 1 else "paragraph"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf_structured(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a PDF into typed elements using PyMuPDF block extraction.

    Each element dict contains:
      element_type  — paragraph | figure
      text          — extracted text (or placeholder for figure/image pages)
      section_path  — "Page N"
      page_or_sheet — str(page_number)
      html_or_markdown — ""
      is_image_page — True only on image-heavy pages (informational flag)

    Warnings are logged for image-heavy pages where OCR would improve results.
    """
    logger.info("[PDF] Parsing structured: %s", file_path.name)

    elements: List[Dict[str, Any]] = []
    image_heavy_pages: List[int] = []

    try:
        with fitz.open(str(file_path)) as pdf:
            total_pages = len(pdf)

            for page_num, page in enumerate(pdf, 1):
                page_text = page.get_text().strip()
                images = page.get_images()
                is_image_heavy = (len(images) > 0 and len(page_text) < _IMAGE_PAGE_CHAR_THRESHOLD)

                if is_image_heavy:
                    image_heavy_pages.append(page_num)
                    elements.append({
                        "element_type": "figure",
                        "text": (
                            f"[Page {page_num}: image/diagram page — "
                            "text content sparse. OCR recommended for full extraction.]"
                        ),
                        "section_path": f"Page {page_num}",
                        "page_or_sheet": str(page_num),
                        "html_or_markdown": "",
                        "is_image_page": True,
                    })
                    continue

                # text="blocks" returns list of (x0,y0,x1,y1,text,block_no,block_type)
                for block in page.get_text("blocks"):
                    text = block[4].strip() if len(block) > 4 else ""
                    if not text:
                        continue
                    etype = _block_element_type(block)
                    elements.append({
                        "element_type": etype,
                        "text": text if etype == "paragraph" else f"[Figure on page {page_num}]",
                        "section_path": f"Page {page_num}",
                        "page_or_sheet": str(page_num),
                        "html_or_markdown": "",
                        "is_image_page": False,
                    })

    except Exception as exc:
        raise ValueError(f"Error parsing PDF '{file_path.name}': {exc}") from exc

    if image_heavy_pages:
        sample = image_heavy_pages[:5]
        more = f"… +{len(image_heavy_pages) - 5} more" if len(image_heavy_pages) > 5 else ""
        logger.warning(
            "[PDF] %s: %d image-heavy page(s) detected (pages %s%s). "
            "Text on those pages is minimal. "
            "For better results, consider an OCR-enabled parser.",
            file_path.name, len(image_heavy_pages), sample, more,
        )

    logger.info("[PDF] %s: %d elements from %d pages", file_path.name, len(elements), total_pages)
    return elements
