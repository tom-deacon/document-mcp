"""Structured DOCX parser — headings, tables, lists, and embedded images."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    _W_P      = qn("w:p")
    _W_TBL    = qn("w:tbl")
    _W_NUM_PR = qn("w:numPr")
    _W_PPR    = qn("w:pPr")
    _W_DRAWING = qn("w:drawing")   # container for inline/anchored drawings
    _A_BLIP    = qn("a:blip")      # DrawingML image reference
    _R_EMBED   = qn("r:embed")     # relationship ID attribute
    _WP_DOC_PR = qn("wp:docPr")    # document properties (alt text / title)

    DOCX_AVAILABLE = True
except ImportError:  # pragma: no cover
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available — structured DOCX parsing disabled")


# ---------------------------------------------------------------------------
# Internal helpers — text structure
# ---------------------------------------------------------------------------

def _get_heading_level(style_name: str) -> Optional[int]:
    """Return 1-9 for Heading styles, 0 for Title, else None."""
    if not style_name:
        return None
    name = style_name.lower().strip()
    if name == "title":
        return 0
    m = re.match(r"heading\s*(\d)", name)
    if m:
        return int(m.group(1))
    return None


def _is_list_item(para) -> bool:
    """Return True if paragraph uses a list/numbering style."""
    style_name = (para.style.name or "").lower() if para.style else ""
    if "list" in style_name:
        return True
    pPr = para._element.find(_W_PPR)
    if pPr is not None and pPr.find(_W_NUM_PR) is not None:
        return True
    return False


def _iter_block_items(doc):
    """Yield Paragraph and Table objects in document body order."""
    for child in doc.element.body:
        if child.tag == _W_P:
            yield Paragraph(child, doc)
        elif child.tag == _W_TBL:
            yield Table(child, doc)


def _rows_to_markdown(rows: List[List[str]]) -> str:
    """Convert a 2-D list of strings to a Markdown table."""
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    padded = [(r + [""] * (ncols - len(r)))[:ncols] for r in rows]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(esc(c) for c in padded[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * ncols) + " |")
    for row in padded[1:]:
        lines.append("| " + " | ".join(esc(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers — image extraction
# ---------------------------------------------------------------------------

def _para_has_drawing(para_element) -> bool:
    """Return True if the paragraph XML contains any drawing (image)."""
    return para_element.find(".//" + _W_DRAWING) is not None


def _extract_para_images(para_element, doc) -> List[tuple]:
    """
    Extract (image_bytes, alt_text) pairs from all drawings in a paragraph.

    Uses the DrawingML blip relationship.  Returns an empty list if the
    paragraph contains no images, or if the image data cannot be retrieved.
    """
    results = []
    for drawing in para_element.iter(_W_DRAWING):
        # Alt text / description lives in wp:docPr
        doc_pr = drawing.find(".//" + _WP_DOC_PR)
        alt_text = ""
        if doc_pr is not None:
            alt_text = (doc_pr.get("descr", "") or doc_pr.get("title", "")).strip()

        # Image bytes are stored in the part referenced by a:blip r:embed
        for blip in drawing.iter(_A_BLIP):
            r_id = blip.get(_R_EMBED)
            if r_id and r_id in doc.part.related_parts:
                try:
                    img_bytes = doc.part.related_parts[r_id].blob
                    results.append((img_bytes, alt_text))
                except Exception:
                    # Part exists but blob unreadable — include as empty
                    results.append((b"", alt_text))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_docx_structured(
    file_path: Path,
    visual_config=None,          # VisualConfig | None
) -> List[Dict[str, Any]]:
    """
    Parse a DOCX file into a list of typed document elements.

    Text elements (heading_1…heading_9, title, paragraph, list_item, table)
    are unchanged from Phase 1.

    New in Phase 2: embedded images produce ``figure`` elements.
    Each figure captures:
      - alt_text   (the image's accessibility description in Word)
      - caption    (following "Caption"-styled paragraph or "Figure X:" label)
      - ocr_text   (Tesseract OCR if pytesseract is installed; else "")
    """
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx is required for structured DOCX parsing")

    # Lazy import to avoid circular dependency
    from .visual_utils import (
        VisualConfig, build_visual_chunk, is_caption_text,
        ocr_is_available, refresh_figure_text, run_ocr,
    )

    vcfg: VisualConfig = visual_config or VisualConfig()
    do_ocr = vcfg.enable_ocr and ocr_is_available()

    logger.info("[DOCX] Parsing structured: %s", file_path.name)

    doc = DocxDocument(str(file_path))
    elements: List[Dict[str, Any]] = []

    section_stack: List[str] = []
    counts: Dict[str, int] = {
        "heading": 0, "paragraph": 0, "list_item": 0, "table": 0, "figure": 0
    }

    # Track figures whose caption has not yet been seen.
    # A "Caption" paragraph immediately following an image paragraph is
    # attached to the most-recent pending figure.
    pending_figures: List[Dict[str, Any]] = []
    document_title = file_path.stem

    for block in _iter_block_items(doc):

        # ---- TABLE -------------------------------------------------------
        if isinstance(block, Table):
            pending_figures.clear()   # tables break caption association
            rows = [[cell.text.strip() for cell in row.cells] for row in block.rows]
            if not rows:
                continue
            md = _rows_to_markdown(rows)
            plain = "\n".join(" | ".join(row) for row in rows)
            elements.append({
                "element_type": "table",
                "text": plain,
                "section_path": " > ".join(section_stack),
                "page_or_sheet": None,
                "html_or_markdown": md,
            })
            counts["table"] += 1
            continue

        # ---- PARAGRAPH ---------------------------------------------------
        if not isinstance(block, Paragraph):
            continue

        para_elem = block._element
        text = block.text.strip()
        style_name = block.style.name if block.style else ""

        # -- Image paragraph (may have text too, e.g. a labelled diagram) --
        if _para_has_drawing(para_elem):
            img_list = _extract_para_images(para_elem, doc)
            for img_bytes, alt_text in img_list:
                # OCR: skip tiny images (logos, bullets, decorative)
                ocr_text = ""
                if do_ocr and len(img_bytes) >= vcfg.min_ocr_image_bytes:
                    ocr_text = run_ocr(img_bytes, vcfg.ocr_language)

                fig = build_visual_chunk(
                    document_title=document_title,
                    section_path=" > ".join(section_stack),
                    page_or_sheet=None,
                    alt_text=alt_text,
                    caption="",          # filled in below if caption follows
                    ocr_text=ocr_text,
                )
                elements.append(fig)
                pending_figures.append(fig)
                counts["figure"] += 1

            # If the image paragraph also has inline text, treat that text
            # as a likely caption/label and attach it immediately.
            if text and pending_figures:
                for fig in pending_figures:
                    if not fig["caption"]:
                        fig["caption"] = text
                        refresh_figure_text(fig, document_title)
                pending_figures.clear()
            continue

        # -- Caption paragraph (attaches to preceding images) --------------
        is_cap_style = "caption" in style_name.lower()
        is_cap_text  = bool(text) and is_caption_text(text)

        if (is_cap_style or is_cap_text) and pending_figures:
            for fig in pending_figures:
                if not fig["caption"]:
                    fig["caption"] = text
                    refresh_figure_text(fig, document_title)
            pending_figures.clear()
            # Don't emit the caption as a separate chunk — it's already in
            # the figure element.
            continue

        # Any non-caption, non-image paragraph ends the caption window.
        pending_figures.clear()

        if not text:
            continue

        # -- Heading -------------------------------------------------------
        hlevel = _get_heading_level(style_name)
        if hlevel is not None:
            if hlevel == 0:
                section_stack = [text]
            else:
                section_stack = section_stack[:hlevel - 1] + [text]
            etype = "title" if hlevel == 0 else f"heading_{hlevel}"
            elements.append({
                "element_type": etype,
                "text": text,
                "section_path": " > ".join(section_stack),
                "page_or_sheet": None,
                "html_or_markdown": "",
            })
            counts["heading"] += 1
            continue

        # -- List item -----------------------------------------------------
        if _is_list_item(block):
            elements.append({
                "element_type": "list_item",
                "text": text,
                "section_path": " > ".join(section_stack),
                "page_or_sheet": None,
                "html_or_markdown": "",
            })
            counts["list_item"] += 1
            continue

        # -- Regular paragraph ---------------------------------------------
        elements.append({
            "element_type": "paragraph",
            "text": text,
            "section_path": " > ".join(section_stack),
            "page_or_sheet": None,
            "html_or_markdown": "",
        })
        counts["paragraph"] += 1

    ocr_note = " (OCR active)" if do_ocr else " (OCR off)"
    logger.info(
        "[DOCX] %s: %d headings, %d paragraphs, %d list items, "
        "%d tables, %d figures%s",
        file_path.name, counts["heading"], counts["paragraph"],
        counts["list_item"], counts["table"], counts["figure"], ocr_note,
    )
    return elements
