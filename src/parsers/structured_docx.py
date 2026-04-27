"""Structured DOCX parser — preserves heading hierarchy, tables, and lists."""

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

    _W_P = qn("w:p")
    _W_TBL = qn("w:tbl")
    _W_NUM_PR = qn("w:numPr")
    _W_PPR = qn("w:pPr")
    DOCX_AVAILABLE = True
except ImportError:  # pragma: no cover
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available — structured DOCX parsing disabled")


# ---------------------------------------------------------------------------
# Internal helpers
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
# Public API
# ---------------------------------------------------------------------------

def parse_docx_structured(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a DOCX file into a list of typed document elements.

    Each element dict contains:
      element_type  — title | heading_1..heading_9 | paragraph | list_item | table
      text          — plain text content
      section_path  — breadcrumb like "Chapter 1 > Section 1.1"
      page_or_sheet — None (Word doesn't expose reliable page numbers cheaply)
      html_or_markdown — Markdown table string for table elements, else ""
    """
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx is required for structured DOCX parsing")

    logger.info("[DOCX] Parsing structured: %s", file_path.name)

    doc = DocxDocument(str(file_path))
    elements: List[Dict[str, Any]] = []

    # section_stack[i] = heading text at depth i  (0 = top-level)
    section_stack: List[str] = []
    counts: Dict[str, int] = {"heading": 0, "paragraph": 0, "list_item": 0, "table": 0}

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue

            style_name = block.style.name if block.style else ""
            hlevel = _get_heading_level(style_name)

            if hlevel is not None:
                if hlevel == 0:
                    section_stack = [text]
                else:
                    depth = hlevel - 1
                    section_stack = section_stack[:depth] + [text]

                etype = "title" if hlevel == 0 else f"heading_{hlevel}"
                elements.append({
                    "element_type": etype,
                    "text": text,
                    "section_path": " > ".join(section_stack),
                    "page_or_sheet": None,
                    "html_or_markdown": "",
                })
                counts["heading"] += 1

            elif _is_list_item(block):
                elements.append({
                    "element_type": "list_item",
                    "text": text,
                    "section_path": " > ".join(section_stack),
                    "page_or_sheet": None,
                    "html_or_markdown": "",
                })
                counts["list_item"] += 1

            else:
                elements.append({
                    "element_type": "paragraph",
                    "text": text,
                    "section_path": " > ".join(section_stack),
                    "page_or_sheet": None,
                    "html_or_markdown": "",
                })
                counts["paragraph"] += 1

        elif isinstance(block, Table):
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

    logger.info(
        "[DOCX] %s: %d headings, %d paragraphs, %d list items, %d tables",
        file_path.name, counts["heading"], counts["paragraph"],
        counts["list_item"], counts["table"],
    )
    return elements
