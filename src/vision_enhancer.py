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

import asyncio
import base64
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF — already in project dependencies

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL = "claude-sonnet-4-6"
_DPI = 96                    # reduced from 150 — sufficient for legibility, far smaller files
_MAX_TOKENS = 1024
_MAX_IMAGE_BYTES = 4 * 1024 * 1024   # 4 MB hard ceiling (Anthropic limit is 5 MB)

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


def _page_is_landscape(page: fitz.Page) -> bool:
    """Return True if the page is wider than it is tall."""
    return page.rect.width > page.rect.height


def _rasterise_page(page: fitz.Page, dpi: int = _DPI) -> bytes:
    """Render *page* to a JPEG byte string at *dpi* dots-per-inch.

    If the initial render exceeds _MAX_IMAGE_BYTES the pixmap is scaled down
    proportionally (halving dimensions each pass) until it fits.
    """
    from PIL import Image
    import io as _io

    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    # Convert PyMuPDF pixmap → PIL Image for JPEG encoding and resizing
    img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    def _encode_jpeg(image: Image.Image) -> bytes:
        buf = _io.BytesIO()
        image.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()

    jpeg_bytes = _encode_jpeg(img)

    # Proportionally shrink until under the size ceiling
    while len(jpeg_bytes) > _MAX_IMAGE_BYTES:
        new_w = img.width // 2
        new_h = img.height // 2
        logger.debug(
            "[VISION] Image %.1f MB > limit — resizing to %dx%d",
            len(jpeg_bytes) / 1_048_576, new_w, new_h,
        )
        img = img.resize((new_w, new_h), Image.LANCZOS)
        jpeg_bytes = _encode_jpeg(img)

    return jpeg_bytes


def _call_vision_api(image_bytes: bytes, api_key: str) -> str:
    """Send *image_bytes* (JPEG) to the Anthropic vision API and return the description."""
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
                            "media_type": "image/jpeg",
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


async def _call_vision_api_async(image_bytes: bytes, api_key: str) -> str:
    """Async version of _call_vision_api using AsyncAnthropic."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")

    response = await client.messages.create(
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
                            "media_type": "image/jpeg",
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


async def _process_page_async(
    sem: asyncio.Semaphore,
    page_num: int,
    total: int,
    image_bytes: bytes,
    api_key: str,
) -> Tuple[int, Optional[str]]:
    """Acquire the semaphore, call the vision API for one page, return (page_num, description).

    Returns (page_num, None) on failure so the caller can skip without aborting the batch.
    """
    async with sem:
        print(
            f"[VISION] Page {page_num}/{total}: image {len(image_bytes) / 1_048_576:.2f} MB "
            f"({len(image_bytes):,} bytes) — sending to API …",
            file=sys.stderr, flush=True,
        )
        try:
            description = await _call_vision_api_async(image_bytes, api_key)
            return page_num, description
        except Exception as exc:
            logger.warning(
                "[VISION] Page %d: API call failed — %s: %s",
                page_num, type(exc).__name__, exc,
            )
            return page_num, None


# ---------------------------------------------------------------------------
# Description splitting
# ---------------------------------------------------------------------------

def _split_description(
    description: str,
    page_num: int,
    max_chars: int = 800,
) -> List[Dict[str, Any]]:
    """
    Return one or more element dicts for *description*.

    If the description fits within *max_chars*, a single element is returned
    using the existing format.  Otherwise it is split at double-newline
    (paragraph) boundaries so that each chunk's content stays within
    *max_chars*, and each element gets a "part X of Y" prefix and section_path.
    Paragraphs are never split mid-text.
    """
    if len(description) <= max_chars:
        return [{
            "element_type": "vision_description",
            "text": f"[Vision description of slide {page_num}]: {description}",
            "section_path": f"Page {page_num}",
            "page_or_sheet": str(page_num),
            "html_or_markdown": "",
            "extracted_by": "vision",
        }]

    # Split at paragraph boundaries and accumulate into ≤ max_chars chunks.
    paragraphs = [p.strip() for p in description.split("\n\n") if p.strip()]
    raw_chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        # Length added to the running total: separator + paragraph
        addition = len(para) if not current_parts else 2 + len(para)  # 2 = len("\n\n")
        if current_parts and current_len + addition > max_chars:
            raw_chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_len = len(para)
        else:
            current_parts.append(para)
            current_len += addition

    if current_parts:
        raw_chunks.append("\n\n".join(current_parts))

    total = len(raw_chunks)
    elements: List[Dict[str, Any]] = []
    for part_idx, chunk_text in enumerate(raw_chunks, 1):
        elements.append({
            "element_type": "vision_description",
            "text": (
                f"[Vision description of slide {page_num}, "
                f"part {part_idx} of {total}]: {chunk_text}"
            ),
            "section_path": f"Page {page_num} (part {part_idx} of {total})",
            "page_or_sheet": str(page_num),
            "html_or_markdown": "",
            "extracted_by": "vision",
        })
    return elements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enhance_pdf(
    file_path: Path,
    word_threshold: int = 50,
    api_key: Optional[str] = None,
    dpi: int = _DPI,
    mode: str = "threshold",
) -> List[Dict[str, Any]]:
    """
    Generate vision-description elements for visually-rich pages in *file_path*.

    Parameters
    ----------
    file_path       : path to the PDF to analyse
    word_threshold  : pages with fewer words than this are sent to vision
                      (only used when mode='threshold')
    api_key         : Anthropic API key; falls back to ANTHROPIC_API_KEY env var
    dpi             : rasterisation resolution (150 DPI is sufficient for legibility)
    mode            : invocation rule — one of:
                        'threshold'  pages below word_threshold are sent (default)
                        'landscape'  all landscape-orientation pages are sent;
                                     portrait pages are always skipped

    Returns
    -------
    List of element dicts — one per selected page — ready to be merged into
    the PDF parser's element list before chunking.  Each dict has:
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

    # ------------------------------------------------------------------
    # Pass 1 — rasterise qualifying pages (synchronous; fitz is not
    # thread/async safe so all page access must stay in this block).
    # ------------------------------------------------------------------
    pages_to_process: List[Tuple[int, bytes]] = []  # (page_num, jpeg_bytes)
    total = 0

    try:
        with fitz.open(str(file_path)) as pdf:
            total = len(pdf)
            for page_idx in range(total):
                page_num = page_idx + 1  # 1-indexed to match Docling convention
                page = pdf[page_idx]

                if mode == "all":
                    reason = "all-pages mode"
                elif mode == "landscape":
                    if not _page_is_landscape(page):
                        logger.debug(
                            "[VISION] Page %d/%d: portrait — skipping",
                            page_num, total,
                        )
                        continue
                    reason = f"landscape ({page.rect.width:.0f}x{page.rect.height:.0f} pts)"
                else:
                    word_count = _page_word_count(page)
                    if word_count >= word_threshold:
                        logger.debug(
                            "[VISION] Page %d/%d: %d words — text-dense, skipping",
                            page_num, total, word_count,
                        )
                        continue
                    reason = f"{word_count} words < {word_threshold} threshold"

                print(
                    f"[VISION] Page {page_num}/{total}: {reason} — rasterising …",
                    file=sys.stderr, flush=True,
                )

                try:
                    image_bytes = _rasterise_page(page, dpi=dpi)
                    pages_to_process.append((page_num, image_bytes))
                except Exception as exc:
                    logger.warning(
                        "[VISION] Page %d: rasterisation failed — %s: %s",
                        page_num, type(exc).__name__, exc,
                    )

    except Exception as exc:
        logger.error(
            "[VISION] Failed to open/rasterise %s — %s: %s",
            file_path.name, type(exc).__name__, exc,
        )
        return vision_elements

    if not pages_to_process:
        return vision_elements

    # ------------------------------------------------------------------
    # Pass 2 — fire API calls in parallel, capped at 5 concurrent requests.
    # ------------------------------------------------------------------
    print(
        f"[VISION] Dispatching {len(pages_to_process)} page(s) to Claude vision "
        f"(concurrency=5) …",
        file=sys.stderr, flush=True,
    )

    async def _gather_pages() -> List[Tuple[int, Optional[str]]]:
        sem = asyncio.Semaphore(5)
        tasks = [
            _process_page_async(sem, page_num, total, image_bytes, resolved_key)
            for page_num, image_bytes in pages_to_process
        ]
        return await asyncio.gather(*tasks)

    try:
        results: List[Tuple[int, Optional[str]]] = asyncio.run(_gather_pages())
    except RuntimeError:
        # Already inside a running event loop (e.g. Jupyter) — fall back to
        # nest_asyncio or a thread-based workaround isn't available; re-raise
        # with a clear message so the caller knows what happened.
        logger.error(
            "[VISION] asyncio.run() called from within a running event loop. "
            "Call enhance_pdf from synchronous code or use nest_asyncio."
        )
        raise

    # ------------------------------------------------------------------
    # Pass 3 — collect results in page order and build element list.
    # ------------------------------------------------------------------
    results.sort(key=lambda r: r[0])  # ensure page order regardless of arrival

    for page_num, description in results:
        if description is None:
            continue  # already logged in _process_page_async

        new_elements = _split_description(description, page_num)
        vision_elements.extend(new_elements)

        n_parts = len(new_elements)
        part_info = f" → {n_parts} sub-chunk(s)" if n_parts > 1 else ""
        print(
            f"[VISION] Page {page_num}: description generated "
            f"({len(description):,} chars){part_info}",
            file=sys.stderr, flush=True,
        )

    logger.info(
        "[VISION] %s: %d vision-description element(s) generated (across all pages/parts)",
        file_path.name, len(vision_elements),
    )
    return vision_elements
