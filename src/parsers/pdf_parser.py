"""
PDF parser — Docling-first layout-aware extraction with EasyOCR fallback.

Parsing strategy (parse_pdf):
  1. Attempt Docling (layout model, table recognition, figure/caption detection).
  2. For any page that Docling returns no content for, fall back to the
     EasyOCR-based pipeline (PyMuPDF block extraction + image OCR).

All elements carry an ``extracted_by`` field ("docling" or "easyocr") so
downstream metadata and search results show which method was used per chunk.

Public API (called from parser.py via _route_to_parser):
  parse_pdf(file_path, visual_config) -> list[dict]
  parse_pdf_structured(file_path, visual_config) -> list[dict]   # thin alias
"""

import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Pages processed per Docling convert() call.  Smaller = more progress messages
# and better per-batch error isolation; larger = slightly less overhead.
_DOCLING_BATCH_PAGES = 10

# A page is "image-heavy" when it has at least one image but fewer than this
# many characters of selectable text.
_IMAGE_PAGE_CHAR_THRESHOLD = 80

# Minimum pixel area for an image to be worth OCR-ing.
_MIN_IMAGE_PIXELS = 4_000   # e.g. 80 x 50 px


# ---------------------------------------------------------------------------
# Helpers (shared by both Docling and EasyOCR paths)
# ---------------------------------------------------------------------------

def _page_caption_texts(blocks: list) -> List[str]:
    """Return text blocks on a page that look like figure/chart captions."""
    caps = []
    for b in blocks:
        if len(b) < 5:
            continue
        btype = b[6] if len(b) > 6 else 0
        if btype != 0:
            continue
        txt = b[4].strip()
        if txt:
            from .visual_utils import is_caption_text
            if is_caption_text(txt):
                caps.append(txt)
    return caps


def _extract_page_figures(
    pdf: fitz.Document,
    page: fitz.Page,
    page_num: int,
    page_captions: List[str],
    vcfg,
    document_title: str,
) -> List[Dict[str, Any]]:
    """Extract figure chunks for all significant images on *page* using EasyOCR."""
    from .visual_utils import build_visual_chunk, ocr_is_available, run_ocr

    do_ocr = vcfg.enable_ocr and ocr_is_available()
    figures: List[Dict[str, Any]] = []

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

        try:
            base_img = pdf.extract_image(xref)
            img_bytes = base_img.get("image", b"")
        except Exception as exc:
            logger.debug("[PDF] Page %d: could not extract image xref=%d: %s", page_num, xref, exc)
            img_bytes = b""

        ocr_text = ""
        if do_ocr and len(img_bytes) >= vcfg.min_ocr_image_bytes:
            ocr_text = run_ocr(img_bytes, vcfg.ocr_language)
            if ocr_text:
                logger.debug(
                    "[PDF] Page %d image #%d: OCR extracted %d chars",
                    page_num, idx, len(ocr_text),
                )

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
# Docling path
# ---------------------------------------------------------------------------

def _docling_items_to_elements(
    doc: Any,
    elements: List[Dict[str, Any]],
    covered_pages: Set[int],
) -> None:
    """Extract structured elements from a Docling DoclingDocument into *elements*."""
    for item, level in doc.iterate_items():
        page_num: Optional[int] = None
        try:
            if item.prov:
                page_num = int(item.prov[0].page_no)
        except (AttributeError, IndexError, TypeError, ValueError):
            pass

        type_name = type(item).__name__

        if type_name == "SectionHeaderItem":
            elem_type = f"heading_{min(max(level, 1), 9)}"
            text = getattr(item, "text", "") or ""
            md = ""
            caption = ""

        elif type_name == "TableItem":
            elem_type = "table"
            try:
                md = item.export_to_markdown()
                text = md
            except Exception:
                md = ""
                text = getattr(item, "text", "") or ""
            try:
                caption = item.caption_text(doc) or ""
            except Exception:
                caption = ""

        elif type_name == "PictureItem":
            elem_type = "figure"
            try:
                caption = item.caption_text(doc) or ""
            except Exception:
                caption = ""
            label = f"[Figure on page {page_num}]" if page_num else "[Figure]"
            text = f"{label}\nCaption: {caption}" if caption else label
            md = ""

        elif type_name == "ListItem":
            elem_type = "list_item"
            text = getattr(item, "text", "") or ""
            md = ""
            caption = ""

        elif type_name == "TextItem":
            elem_type = "paragraph"
            text = getattr(item, "text", "") or ""
            md = ""
            caption = ""

        else:
            text = getattr(item, "text", "") or ""
            if not text.strip():
                continue
            elem_type = "paragraph"
            md = ""
            caption = ""

        if not text.strip():
            continue

        elem: Dict[str, Any] = {
            "element_type": elem_type,
            "text": text,
            "section_path": f"Page {page_num}" if page_num else "",
            "page_or_sheet": str(page_num) if page_num else "",
            "html_or_markdown": md,
            "extracted_by": "docling",
        }
        if caption:
            elem["caption"] = caption

        elements.append(elem)
        if page_num:
            covered_pages.add(page_num)


def parse_pdf_with_docling(
    file_path: Path,
    document_title: str,
    total_pages: int,
) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """
    Parse a PDF using Docling's layout-aware model.

    Processes the document in batches of _DOCLING_BATCH_PAGES pages so that
    progress is visible and a single bad-page does not abort the whole document.

    Returns:
        elements        — element dicts with extracted_by="docling".
        covered_pages   — 1-indexed page numbers that have at least one element.

    Raises ImportError if docling is not installed (caller catches this).
    """
    def _p(msg: str) -> None:
        """Immediate flushed print — visible even if logging is buffered."""
        safe = msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace")
        print(f"[DOCLING] {safe}", flush=True)

    _p(f"► importing Docling modules  ({file_path.name})")
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415
    from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415
    _p("► Docling modules imported OK")

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.images_scale = 1.0
    pipeline_opts.generate_page_images = False
    pipeline_opts.generate_picture_images = False
    _p("► pipeline options built; creating DocumentConverter …")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )
    _p("► DocumentConverter created (models loaded)")

    elements: List[Dict[str, Any]] = []
    covered_pages: Set[int] = set()
    t_overall = time.monotonic()

    # --- Try page-range batching first -----------------------------------
    batch_supported = True
    batches_done = 0
    batches_total = (total_pages + _DOCLING_BATCH_PAGES - 1) // _DOCLING_BATCH_PAGES

    for batch_start in range(1, total_pages + 1, _DOCLING_BATCH_PAGES):
        batch_end = min(batch_start + _DOCLING_BATCH_PAGES - 1, total_pages)
        batches_done += 1
        _p(f"► calling converter.convert()  pages {batch_start}–{batch_end} / {total_pages}  (batch {batches_done}/{batches_total})")
        t_batch = time.monotonic()
        try:
            result = converter.convert(
                str(file_path), page_range=(batch_start, batch_end)
            )
            _p(f"► converter.convert() returned after {time.monotonic()-t_batch:.1f}s — extracting items")
            _docling_items_to_elements(result.document, elements, covered_pages)
            _p(f"► batch {batches_done}/{batches_total} done  ({len(elements)} elements so far)")
            logger.info(
                "[PDF] Docling: batch %d/%d done in %.1fs  (%d elements so far)",
                batches_done, batches_total,
                time.monotonic() - t_batch, len(elements),
            )
        except TypeError:
            _p("► page_range not supported — switching to single-pass mode")
            logger.info("[PDF] Docling: page_range not supported; switching to single-pass mode")
            batch_supported = False
            elements.clear()
            covered_pages.clear()
            break
        except Exception as exc:
            _p(f"► batch {batches_done} FAILED after {time.monotonic()-t_batch:.1f}s: {type(exc).__name__}: {exc}")
            logger.warning(
                "[PDF] Docling: batch pages %d–%d failed after %.1fs: %s: %s — "
                "those pages will use EasyOCR fallback.",
                batch_start, batch_end,
                time.monotonic() - t_batch,
                type(exc).__name__, exc,
            )

    # --- Fallback: single blocking call with heartbeat thread ------------
    if not batch_supported:
        _stop = threading.Event()

        def _heartbeat() -> None:
            t0 = time.monotonic()
            while not _stop.wait(30):
                _p(f"still processing… ({time.monotonic()-t0:.0f}s elapsed)")

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()
        _p(f"► calling converter.convert() single-pass on {file_path.name}")
        try:
            result = converter.convert(str(file_path))
            _p(f"► converter.convert() returned after {time.monotonic()-t_overall:.1f}s")
            _docling_items_to_elements(result.document, elements, covered_pages)
        except Exception as exc:
            _p(f"► single-pass FAILED: {type(exc).__name__}: {exc}")
            logger.error(
                "[PDF] Docling single-pass FAILED after %.1fs: %s: %s",
                time.monotonic() - t_overall, type(exc).__name__, exc,
            )
            raise
        finally:
            _stop.set()
            hb.join(timeout=1)

    logger.info(
        "[PDF] Docling complete: %d elements across %d page(s) in %.1fs  (%s)",
        len(elements), len(covered_pages),
        time.monotonic() - t_overall, file_path.name,
    )
    return elements, covered_pages


# ---------------------------------------------------------------------------
# EasyOCR path (for specific pages only)
# ---------------------------------------------------------------------------

def _parse_pages_with_easyocr(
    file_path: Path,
    pages: Set[int],
    vcfg,
    document_title: str,
) -> List[Dict[str, Any]]:
    """
    Run the EasyOCR-based pipeline on a specific subset of pages.

    Replicates the per-page logic from the original parse_pdf_structured(),
    but restricted to the given page numbers and with extracted_by="easyocr".
    """
    elements: List[Dict[str, Any]] = []
    try:
        with fitz.open(str(file_path)) as pdf:
            total = len(pdf)
            for page_num in sorted(pages):
                if page_num < 1 or page_num > total:
                    continue

                page = pdf[page_num - 1]  # fitz is 0-indexed
                page_text = page.get_text().strip()
                raw_blocks = page.get_text("blocks")
                image_list = page.get_images()
                is_image_heavy = (
                    len(image_list) > 0
                    and len(page_text) < _IMAGE_PAGE_CHAR_THRESHOLD
                )
                page_captions = _page_caption_texts(raw_blocks)

                # Extract images via EasyOCR
                page_figures = _extract_page_figures(
                    pdf, page, page_num, page_captions, vcfg, document_title
                )
                for fig in page_figures:
                    fig["extracted_by"] = "easyocr"
                elements.extend(page_figures)

                if is_image_heavy and not page_figures:
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
                        "extracted_by": "easyocr",
                    })
                    continue  # skip text block pass for image-heavy pages

                # Text block pass for normal pages
                for block in raw_blocks:
                    if len(block) < 5:
                        continue
                    btype = block[6] if len(block) > 6 else 0
                    text = block[4].strip() if len(block) > 4 else ""
                    if not text or btype == 1:
                        continue
                    elements.append({
                        "element_type": "paragraph",
                        "text": text,
                        "section_path": f"Page {page_num}",
                        "page_or_sheet": str(page_num),
                        "html_or_markdown": "",
                        "is_image_page": False,
                        "extracted_by": "easyocr",
                    })

    except Exception as exc:
        logger.warning("[PDF] EasyOCR fallback failed for %s: %s", file_path.name, exc)

    return elements


# ---------------------------------------------------------------------------
# Public routing function
# ---------------------------------------------------------------------------

def parse_pdf(
    file_path: Path,
    visual_config=None,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Parse a PDF using Docling first, with EasyOCR fallback for uncovered pages.

    Strategy:
      1. If Docling is installed, convert the whole PDF with Docling.
         Track which pages have content.
      2. Any page with no Docling content gets EasyOCR treatment.
      3. If Docling raises an exception or is not installed, all pages use EasyOCR.

    max_pages: if set, only the first N pages are processed (useful for testing).

    All returned elements have an ``extracted_by`` field:
      "docling"  — element came from the Docling layout model
      "easyocr"  — element came from the original PyMuPDF + EasyOCR pipeline

    Return format is identical to the old parse_pdf_structured() so the rest
    of the pipeline (chunker, indexer) requires no structural changes.
    """
    from .visual_utils import VisualConfig
    vcfg: VisualConfig = visual_config or VisualConfig()
    document_title = file_path.stem

    # Count total pages (needed for fallback page range)
    total_pages = 0
    try:
        with fitz.open(str(file_path)) as pdf:
            total_pages = len(pdf)
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{file_path.name}': {exc}") from exc

    if max_pages is not None:
        total_pages = min(total_pages, max_pages)
        logger.info("[PDF] max_pages=%d — processing pages 1–%d only", max_pages, total_pages)

    all_page_nums: Set[int] = set(range(1, total_pages + 1))

    # ------------------------------------------------------------------
    # Step 1 — Attempt Docling
    # ------------------------------------------------------------------
    docling_elements: List[Dict[str, Any]] = []
    covered_pages: Set[int] = set()

    print(f"[PDF] parse_pdf() reached — checking Docling availability", flush=True)
    try:
        import docling  # noqa: F401  — check availability without full import
        docling_ok = True
        print(f"[PDF] Docling import OK — will attempt Docling path", flush=True)
    except ImportError:
        logger.info(
            "[PDF] Docling not installed — using EasyOCR pipeline for all pages "
            "(%s). Install with: uv add docling",
            file_path.name,
        )
        docling_ok = False

    if docling_ok:
        try:
            print(f"[PDF] calling parse_pdf_with_docling() for {file_path.name}", flush=True)
            docling_elements, covered_pages = parse_pdf_with_docling(
                file_path, document_title, total_pages
            )
        except Exception as exc:
            logger.warning(
                "[PDF] Docling failed for '%s' — falling back to EasyOCR for all pages.\n"
                "  Error: %s: %s",
                file_path.name, type(exc).__name__, exc,
            )
            docling_elements = []
            covered_pages = set()

    # ------------------------------------------------------------------
    # Step 2 — EasyOCR fallback for pages Docling did not cover
    # ------------------------------------------------------------------
    fallback_pages = all_page_nums - covered_pages
    fallback_elements: List[Dict[str, Any]] = []

    if fallback_pages:
        if covered_pages:
            logger.info(
                "[PDF] EasyOCR fallback for %d uncovered page(s): %s%s",
                len(fallback_pages),
                sorted(fallback_pages)[:10],
                " …" if len(fallback_pages) > 10 else "",
            )
        else:
            logger.info(
                "[PDF] EasyOCR pipeline: all %d page(s) of %s",
                total_pages, file_path.name,
            )
        fallback_elements = _parse_pages_with_easyocr(
            file_path, fallback_pages, vcfg, document_title
        )

    # ------------------------------------------------------------------
    # Step 3 — Merge and sort by page number
    # ------------------------------------------------------------------
    all_elements = docling_elements + fallback_elements

    def _page_sort_key(e: Dict[str, Any]) -> int:
        try:
            return int(e.get("page_or_sheet") or 0)
        except (ValueError, TypeError):
            return 0

    all_elements.sort(key=_page_sort_key)

    # Summary logging
    docling_count  = sum(1 for e in all_elements if e.get("extracted_by") == "docling")
    easyocr_count  = sum(1 for e in all_elements if e.get("extracted_by") == "easyocr")
    logger.info(
        "[PDF] %s: %d total elements — %d docling, %d easyocr",
        file_path.name, len(all_elements), docling_count, easyocr_count,
    )

    return all_elements


# ---------------------------------------------------------------------------
# Backward-compatible alias (called from parser.py via _route_to_parser)
# ---------------------------------------------------------------------------

def parse_pdf_structured(
    file_path: Path,
    visual_config=None,
) -> List[Dict[str, Any]]:
    """Thin alias for parse_pdf() — preserves the existing public interface."""
    return parse_pdf(file_path, visual_config=visual_config)
