"""
Enhanced PDF parser — block-level text elements + per-page image extraction.

Phase 1: classified text blocks, flagged image-heavy pages.
Phase 2 additions:
  - extracts actual image bytes from every page (not just image-heavy pages)
  - runs OCR on images that are large enough to contain meaningful text
  - scans for caption-style text blocks near each image
  - produces proper figure chunks instead of placeholder strings
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# A page is "image-heavy" when it has at least one image but fewer than this
# many characters of selectable text.
_IMAGE_PAGE_CHAR_THRESHOLD = 80

# Minimum pixel area for an image to be worth OCR-ing.
# Filters out logos, icons, decorative bullets, watermarks.
_MIN_IMAGE_PIXELS = 4_000   # e.g. 80 x 50 px


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page_caption_texts(blocks: list) -> List[str]:
    """
    Return any text blocks on a page that look like figure/chart captions.
    Blocks is the output of page.get_text('blocks').
    """
    caps = []
    for b in blocks:
        if len(b) < 5:
            continue
        btype = b[6] if len(b) > 6 else 0
        if btype != 0:
            continue
        txt = b[4].strip()
        if txt:
            # Import lazily to avoid circular import at module load
            from .visual_utils import is_caption_text
            if is_caption_text(txt):
                caps.append(txt)
    return caps


def _extract_page_figures(
    pdf: fitz.Document,
    page: fitz.Page,
    page_num: int,
    page_captions: List[str],
    vcfg,             # VisualConfig
    document_title: str,
) -> List[Dict[str, Any]]:
    """
    Extract figure chunks for all significant images on *page*.

    For each image that meets the size threshold:
      - run OCR (if enabled and pytesseract available)
      - attach a page-level caption if one was found on this page
      - call build_visual_chunk to produce a standardised chunk
    """
    from .visual_utils import build_visual_chunk, ocr_is_available, run_ocr

    do_ocr = vcfg.enable_ocr and ocr_is_available()
    figures: List[Dict[str, Any]] = []

    # get_images(full=True) returns:
    # (xref, smask, width, height, bpc, colorspace, alt, name, filter, referencer)
    image_list = page.get_images(full=True)

    for idx, img_info in enumerate(image_list):
        if idx >= vcfg.max_images_per_page:
            logger.debug("[PDF] Page %d: reached max_images_per_page limit", page_num)
            break

        xref   = img_info[0]
        width  = img_info[2]
        height = img_info[3]

        if width * height < _MIN_IMAGE_PIXELS:
            logger.debug(
                "[PDF] Page %d image #%d: too small (%dx%d), skipping",
                page_num, idx, width, height,
            )
            continue

        # Extract raw image bytes
        try:
            base_img = pdf.extract_image(xref)
            img_bytes = base_img.get("image", b"")
        except Exception as exc:
            logger.debug("[PDF] Page %d: could not extract image xref=%d: %s", page_num, xref, exc)
            img_bytes = b""

        # OCR
        ocr_text = ""
        if do_ocr and len(img_bytes) >= vcfg.min_ocr_image_bytes:
            ocr_text = run_ocr(img_bytes, vcfg.ocr_language)
            if ocr_text:
                logger.debug(
                    "[PDF] Page %d image #%d: OCR extracted %d chars",
                    page_num, idx, len(ocr_text),
                )

        # Caption: use the first page-level caption (best effort)
        caption = page_captions[0] if page_captions else ""

        fig = build_visual_chunk(
            document_title=document_title,
            section_path=f"Page {page_num}",
            page_or_sheet=str(page_num),
            alt_text="",
            caption=caption,
            ocr_text=ocr_text,
            figure_label=f"Image {idx + 1} on page {page_num}",
            element_type="figure",
        )
        figures.append(fig)

    return figures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf_structured(
    file_path: Path,
    visual_config=None,     # VisualConfig | None
) -> List[Dict[str, Any]]:
    """
    Parse a PDF into typed elements using PyMuPDF block extraction.

    Text elements:
      element_type = paragraph  (text blocks)

    Visual elements (Phase 2):
      element_type = figure     (one per significant image on a page)
        - text includes: label, caption (if detected), OCR text
        - embedding_text is enriched with document + page + alt/caption/OCR

    Image-heavy pages (few chars, one or more images) are fully handled:
    their images are extracted and OCR-ed rather than left as placeholders.

    Logging:
      - counts figures detected per page
      - warns when a page has images but OCR produced no text
      - summarises image-heavy pages at the end
    """
    from .visual_utils import VisualConfig

    vcfg: VisualConfig = visual_config or VisualConfig()

    logger.info("[PDF] Parsing structured: %s", file_path.name)

    elements: List[Dict[str, Any]] = []
    image_heavy_pages: List[int] = []
    total_figures = 0

    try:
        with fitz.open(str(file_path)) as pdf:
            total_pages = len(pdf)

            for page_num, page in enumerate(pdf, 1):
                page_text = page.get_text().strip()
                raw_blocks = page.get_text("blocks")
                image_list = page.get_images()
                is_image_heavy = (
                    len(image_list) > 0
                    and len(page_text) < _IMAGE_PAGE_CHAR_THRESHOLD
                )

                # Collect captions from this page's text blocks
                page_captions = _page_caption_texts(raw_blocks)

                # ---- Extract figure chunks for this page ----
                page_figures = _extract_page_figures(
                    pdf, page, page_num, page_captions, vcfg,
                    document_title=file_path.stem,
                )
                if page_figures:
                    elements.extend(page_figures)
                    total_figures += len(page_figures)
                    logger.debug(
                        "[PDF] Page %d: %d figure(s) extracted", page_num, len(page_figures)
                    )

                if is_image_heavy:
                    image_heavy_pages.append(page_num)
                    # The figures were already added above.
                    # If no figures were found (e.g. all images too small),
                    # add a placeholder so the page is at least represented.
                    if not page_figures:
                        elements.append({
                            "element_type": "figure",
                            "text": (
                                f"[Page {page_num}: image/diagram page - "
                                "text content sparse]"
                            ),
                            "section_path": f"Page {page_num}",
                            "page_or_sheet": str(page_num),
                            "html_or_markdown": "",
                            "is_image_page": True,
                            "alt_text": "",
                            "caption": page_captions[0] if page_captions else "",
                            "ocr_text": "",
                        })
                    continue   # skip text-block pass for image-heavy pages

                # ---- Text block pass for normal pages ----
                for block in raw_blocks:
                    if len(block) < 5:
                        continue
                    btype = block[6] if len(block) > 6 else 0
                    text = block[4].strip() if len(block) > 4 else ""
                    if not text:
                        continue

                    if btype == 1:
                        # Image block in a text-rich page — already handled above
                        # via _extract_page_figures; skip placeholder here.
                        continue

                    elements.append({
                        "element_type": "paragraph",
                        "text": text,
                        "section_path": f"Page {page_num}",
                        "page_or_sheet": str(page_num),
                        "html_or_markdown": "",
                        "is_image_page": False,
                    })

    except Exception as exc:
        raise ValueError(f"Error parsing PDF '{file_path.name}': {exc}") from exc

    # ---- Summary logging ----
    if image_heavy_pages:
        sample = image_heavy_pages[:5]
        more = (
            f" +{len(image_heavy_pages) - 5} more"
            if len(image_heavy_pages) > 5 else ""
        )
        logger.warning(
            "[PDF] %s: %d image-heavy page(s) (pages %s%s). "
            "Selectable text is sparse on those pages.",
            file_path.name, len(image_heavy_pages), sample, more,
        )

    if total_figures > 0:
        logger.info(
            "[PDF] %s: %d figure chunk(s) created across %d pages",
            file_path.name, total_figures, total_pages,
        )
    else:
        logger.info(
            "[PDF] %s: no significant images found (%d pages scanned)",
            file_path.name, total_pages,
        )

    logger.info(
        "[PDF] %s: %d total elements from %d pages",
        file_path.name, len(elements), total_pages,
    )
    return elements
