#!/usr/bin/env python3
"""
Standalone ingestion inspection script.

Usage
-----
  python tests/test_ingestion.py path/to/file.docx
  python tests/test_ingestion.py path/to/spreadsheet.xlsx
  python tests/test_ingestion.py path/to/document.pdf
  python tests/test_ingestion.py path/to/file.docx --plain   # force plain-text mode

The script runs the full parse → chunk pipeline and prints every chunk
with its type, section path, token count, and a text preview.  No
database or network connection required.
"""

import argparse
import logging
import sys
import textwrap
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Show INFO-level logs from our parsers and from Docling so progress is visible.
# Filter out the repetitive torch pin_memory noise.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("easyocr").setLevel(logging.WARNING)

from src.parser import DocumentParser


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_TYPE_LABEL = {
    "paragraph":    "PARA ",
    "list_item":    "LIST ",
    "table":        "TABLE",
    "sheet":        "SHEET",
    "figure":       "FIG  ",
    "title":        "TITLE",
}

def _label(etype: str) -> str:
    if etype.startswith("heading_"):
        level = etype.split("_")[-1]
        return f"H{level}   "
    return _TYPE_LABEL.get(etype, etype[:5].upper())


def _bar(n: int, max_n: int = 800, width: int = 20) -> str:
    filled = int(width * min(n, max_n) / max_n)
    return "█" * filled + "░" * (width - filled)


def _preview(text: str, max_chars: int = 120) -> str:
    single = " ".join(text.split())
    return (single[:max_chars] + "…") if len(single) > max_chars else single


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect structured ingestion output for a single file."
    )
    parser.add_argument("file", help="Path to the file to parse")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Use plain-text mode (disables structured parsing)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Token budget per chunk (default 512)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR even if EasyOCR is available",
    )
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="OCR language code (default: eng)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Only process the first N pages (PDF only; useful for quick smoke tests)",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    doc_parser = DocumentParser(
        use_structured=not args.plain,
        max_chunk_tokens=args.max_tokens,
        enable_ocr=not args.no_ocr,
        ocr_language=args.ocr_lang,
        max_pages=args.max_pages,
    )

    from src.parsers.visual_utils import ocr_is_available

    ocr_status = "on" if (not args.no_ocr and ocr_is_available()) else "off"
    if not args.no_ocr and not ocr_is_available():
        ocr_status = "off (EasyOCR not available)"

    print(f"\n{'='*70}")
    print(f"  File     : {file_path.name}")
    print(f"  Mode     : {'plain-text' if args.plain else 'structured'}")
    print(f"  Tokens   : ≤{args.max_tokens} per chunk")
    print(f"  OCR      : {ocr_status}")
    if args.max_pages:
        print(f"  Max pages: {args.max_pages}  (smoke-test mode)")
    print(f"{'='*70}\n")

    try:
        result = doc_parser.parse_file(file_path)
    except Exception as exc:
        print(f"PARSE ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    chunks = result["chunks"]
    meta   = result["metadata"]

    # Summary header
    print(f"  File type    : {meta['file_type']}")
    print(f"  File size    : {meta['file_size'] / 1024:.1f} KB")
    print(f"  Total chars  : {result['total_chars']:,}")
    print(f"  Total tokens : {result['total_tokens']:,}")
    print(f"  Chunks       : {result['num_chunks']}")
    print()

    # Type breakdown
    type_counts: dict = {}
    by_counts: dict = {}
    token_total = 0
    for c in chunks:
        etype = c.get("element_type", "paragraph")
        type_counts[etype] = type_counts.get(etype, 0) + 1
        token_total += c.get("token_count", 0)
        by = c.get("extracted_by", "")
        if by:
            by_counts[by] = by_counts.get(by, 0) + 1

    if type_counts:
        print("  Chunk breakdown:")
        for etype, count in sorted(type_counts.items()):
            print(f"    {_label(etype)}  {count:4d}  chunks")
        avg_tok = token_total // len(chunks) if chunks else 0
        print(f"\n  Average tokens/chunk : {avg_tok}")
    if by_counts:
        print(f"\n  Extracted by:")
        for method, count in sorted(by_counts.items()):
            print(f"    {method:<12}  {count:4d}  chunks")
    print(f"\n{'─'*70}\n")

    # Per-chunk listing
    for i, chunk in enumerate(chunks):
        etype   = chunk.get("element_type", "paragraph")
        section = chunk.get("section_path", "")
        page    = chunk.get("page_or_sheet", "")
        tokens  = chunk.get("token_count", 0)
        text    = chunk.get("text", "")
        emb_txt = chunk.get("embedding_text", "")
        has_md  = bool(chunk.get("html_or_markdown", ""))

        extracted_by = chunk.get("extracted_by", "")
        label   = _label(etype)
        loc     = page if page else (f"§ {section[:40]}" if section else "")
        bar     = _bar(tokens)
        by_tag  = f"  [{extracted_by}]" if extracted_by else ""

        print(f"  [{i:03d}] {label}  {bar}  {tokens:4d} tok  {loc}{by_tag}")
        if section:
            wrapped_section = textwrap.shorten(section, width=65, placeholder="…")
            print(f"        section : {wrapped_section}")
        print(f"        text    : {_preview(text)}")
        if emb_txt and emb_txt != text:
            print(f"        embed↑  : {_preview(emb_txt, 100)}")
        if has_md:
            print(f"        markdown: {_preview(chunk['html_or_markdown'], 100)}")
        # Visual chunk fields (figures only)
        if etype == "figure":
            alt  = chunk.get("alt_text", "")
            cap  = chunk.get("caption", "")
            ocr  = chunk.get("ocr_text", "")
            if alt:
                print(f"        alt_text: {_preview(alt, 100)}")
            if cap:
                print(f"        caption : {_preview(cap, 100)}")
            if ocr:
                print(f"        ocr_text: {_preview(ocr, 100)}")
            elif not alt and not cap:
                print(f"        (no alt text, caption, or OCR text)")
        print()

    print(f"{'='*70}")
    print(f"  Done.  {len(chunks)} chunks from {file_path.name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
