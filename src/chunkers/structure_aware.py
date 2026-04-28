"""
Structure-aware, token-budget chunker.

Rules
-----
- Tables / sheets / figures  -> always their own chunk (standalone).
- Headings                   -> flush the current buffer, then seed a new one.
- Paragraphs / list items    -> merge into the current buffer until the token
                               budget is reached, then flush.
- Oversized single elements  -> split at sentence boundaries.

Each output chunk carries an `embedding_text` field that prepends document
title and section breadcrumb so the vector captures context, not just text.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import tiktoken

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Type categories
# -----------------------------------------------------------------------
_STANDALONE = {"table", "sheet", "figure", "cell_range"}
_HEADING_RE = re.compile(r"^heading_\d$|^title$")


def _is_heading(etype: str) -> bool:
    return bool(_HEADING_RE.match(etype))


# -----------------------------------------------------------------------
# Embedding-text builder
# -----------------------------------------------------------------------

def _build_embedding_text(
    document_title: str,
    section_path: str,
    element_type: str,
    text: str,
    extra: str = "",
) -> str:
    """Assemble a rich string that will be sent to the embedding model."""
    parts: List[str] = []
    if document_title:
        parts.append(f"Document: {document_title}")
    if section_path:
        parts.append(f"Section: {section_path}")
    if element_type in _STANDALONE:
        parts.append(f"Type: {element_type}")
    if extra:
        parts.append(extra)
    # Limit raw text to avoid over-long embedding inputs
    parts.append(text[:2000])
    return " | ".join(parts)


# -----------------------------------------------------------------------
# Sentence splitter (used when a single element exceeds the token budget)
# -----------------------------------------------------------------------

def _split_by_sentences(
    text: str,
    max_tokens: int,
    tokenizer,
    document_title: str,
    section_path: str,
    element_type: str,
    page_or_sheet: str,
) -> List[Dict[str, Any]]:
    """Split a large text block into sentence-boundary chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[Dict[str, Any]] = []
    current: List[str] = []
    current_tokens = 0

    def _flush(sents: List[str], tok_count: int) -> None:
        if not sents:
            return
        chunk_text = " ".join(sents)
        embedding_text = _build_embedding_text(
            document_title, section_path, element_type, chunk_text
        )
        chunks.append({
            "element_type": element_type,
            "text": chunk_text,
            "section_path": section_path,
            "page_or_sheet": page_or_sheet,
            "embedding_text": embedding_text,
            "html_or_markdown": "",
            "token_count": tok_count,
            "char_count": len(chunk_text),
        })

    for sent in sentences:
        t = len(tokenizer.encode(sent))
        if current_tokens + t > max_tokens and current:
            _flush(current, current_tokens)
            current, current_tokens = [sent], t
        else:
            current.append(sent)
            current_tokens += t

    _flush(current, current_tokens)
    return chunks


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def chunk_structured_elements(
    elements: List[Dict[str, Any]],
    document_title: str,
    max_tokens: int = 512,
    tokenizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a list of typed document elements into embedding-ready chunks.

    Parameters
    ----------
    elements       : output of any parse_*_structured() function
    document_title : used to enrich embedding_text (typically the filename)
    max_tokens     : target upper bound for mergeable chunks
    tokenizer      : tiktoken encoder; uses cl100k_base if not supplied

    Returns
    -------
    List of chunk dicts, each with:
      chunk_id, element_type, text, section_path, page_or_sheet,
      embedding_text, html_or_markdown, token_count, char_count,
      start_pos (-1 for structured chunks), end_pos (-1)
    """
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    chunks: List[Dict[str, Any]] = []

    # --- buffer for mergeable elements (paragraphs, list items, headings) ---
    buf_elems: List[Dict] = []
    buf_tokens: int = 0

    def _count(txt: str) -> int:
        return len(tokenizer.encode(txt))

    def _flush_buffer() -> None:
        nonlocal buf_elems, buf_tokens
        if not buf_elems:
            return

        merged_text = "\n\n".join(e["text"] for e in buf_elems)
        section_path = buf_elems[-1].get("section_path", "")
        page_or_sheet = buf_elems[-1].get("page_or_sheet") or ""

        etypes = {e["element_type"] for e in buf_elems}
        # Prefer the most specific heading label; fall back to list_item, then paragraph
        if heading_types := [t for t in etypes if _is_heading(t)]:
            primary_type = sorted(heading_types)[0]
        elif "list_item" in etypes:
            primary_type = "list_item"
        else:
            primary_type = "paragraph"

        # Collect unique extracted_by values from the merged elements
        extracted_by_vals = list(dict.fromkeys(
            e["extracted_by"] for e in buf_elems if e.get("extracted_by")
        ))
        extracted_by = ",".join(extracted_by_vals)

        total_tokens = _count(merged_text)

        if total_tokens <= max_tokens:
            embedding_text = _build_embedding_text(
                document_title, section_path, primary_type, merged_text
            )
            chunks.append({
                "element_type": primary_type,
                "text": merged_text,
                "section_path": section_path,
                "page_or_sheet": page_or_sheet,
                "embedding_text": embedding_text,
                "html_or_markdown": "",
                "token_count": total_tokens,
                "char_count": len(merged_text),
                "start_pos": -1,
                "end_pos": -1,
                "extracted_by": extracted_by,
            })
        else:
            # Element is too large — split by sentences
            sub = _split_by_sentences(
                merged_text, max_tokens, tokenizer,
                document_title, section_path, primary_type, page_or_sheet,
            )
            for c in sub:
                c["start_pos"] = -1
                c["end_pos"] = -1
                c["extracted_by"] = extracted_by
            chunks.extend(sub)

        buf_elems.clear()
        buf_tokens = 0

    # --- main loop --------------------------------------------------------
    for elem in elements:
        etype = elem.get("element_type", "paragraph")
        text = elem.get("text", "").strip()
        if not text:
            continue

        if etype in _STANDALONE:
            # Flush pending buffer first, then emit standalone chunk
            _flush_buffer()

            section_path = elem.get("section_path", "")
            page_or_sheet = elem.get("page_or_sheet") or ""
            md = elem.get("html_or_markdown", "")
            summary = elem.get("summary", "")
            extra = f"Summary: {summary}" if summary else ""

            # Prefer the markdown/table representation for embedding quality
            embed_content = md if md else text
            embedding_text = _build_embedding_text(
                document_title, section_path, etype, embed_content, extra
            )
            chunks.append({
                "element_type": etype,
                "text": text,
                "section_path": section_path,
                "page_or_sheet": page_or_sheet,
                "embedding_text": embedding_text,
                "html_or_markdown": md,
                "token_count": _count(text),
                "char_count": len(text),
                "start_pos": -1,
                "end_pos": -1,
                "extracted_by": elem.get("extracted_by", ""),
            })

        elif _is_heading(etype):
            # Headings flush the buffer, then seed a new one
            _flush_buffer()
            tok = _count(text)
            buf_elems.append(elem)
            buf_tokens += tok

        else:
            # Paragraph / list_item: merge into buffer
            tok = _count(text)
            if buf_tokens + tok > max_tokens and buf_elems:
                _flush_buffer()
            buf_elems.append(elem)
            buf_tokens += tok

    _flush_buffer()

    # Assign sequential chunk IDs
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = i

    # Log chunk type breakdown
    type_counts: Dict[str, int] = {}
    total_tokens = 0
    for c in chunks:
        etype = c["element_type"]
        type_counts[etype] = type_counts.get(etype, 0) + 1
        total_tokens += c.get("token_count", 0)

    avg_tokens = total_tokens // len(chunks) if chunks else 0
    logger.info(
        "[CHUNKER] '%s': %d chunks | %s | avg %d tokens/chunk",
        document_title, len(chunks),
        ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())),
        avg_tokens,
    )

    return chunks
