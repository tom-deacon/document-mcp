#!/usr/bin/env python3
"""Detect and unpack a PDF portfolio (package/collection).

A PDF portfolio bundles multiple files inside a single container PDF using
the /Collection catalog entry.  Files are stored as /Filespec xref objects
with /EF (embedded file) stream references.  Standard PDF parsers only see
the cover page, hence the very low chunk counts from the indexer.

Usage:
    uv run python tools/unpack_pdf_portfolio.py "path/to/file.pdf"
    uv run python tools/unpack_pdf_portfolio.py "path/to/file.pdf" --list-only

Output files are written to a subfolder next to the source file, named
after the source file without its extension.
"""

import re
import sys
import argparse
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF is not installed.  Run: pip install pymupdf")
    sys.exit(1)


# ---------------------------------------------------------------------------
# PDF literal-string helpers
# ---------------------------------------------------------------------------

def _parse_pdf_literal_string(obj: str, key: str) -> str | None:
    """Extract a PDF literal string value for `key` from a PDF object string.

    Handles balanced unescaped parentheses and backslash-escaped chars
    (e.g.  ``\\(``, ``\\)``) which a naïve ``[^)]+`` regex would mangle.
    Returns the decoded (unescaped) string, or None if the key is absent.
    """
    # Match:  /KEY WS* ( ... ) where the inner group handles \. escapes
    pattern = re.compile(
        r"/" + re.escape(key) + r"\s*\(([^)\\]*(?:\\.[^)\\]*)*)\)"
    )
    m = pattern.search(obj)
    if not m:
        return None
    raw = m.group(1)
    # Unescape PDF backslash sequences relevant to filenames
    raw = raw.replace("\\(", "(")
    raw = raw.replace("\\)", ")")
    raw = raw.replace("\\\\", "\\")
    return raw


def _extract_xref_ref(obj: str, key: str) -> int | None:
    """Return the xref number for an indirect reference  /KEY N G R."""
    m = re.search(r"/" + re.escape(key) + r"\s+(\d+)\s+\d+\s+R", obj)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Portfolio detection
# ---------------------------------------------------------------------------

def is_pdf_portfolio(doc: fitz.Document) -> bool:
    """Return True if the PDF catalog contains a /Collection entry."""
    xref = doc.pdf_catalog()
    return "/Collection" in doc.xref_object(xref, compressed=False)


# ---------------------------------------------------------------------------
# Filespec enumeration  (xref-based — works where embfile_* API returns 0)
# ---------------------------------------------------------------------------

def _iter_filespecs(doc: fitz.Document):
    """Yield (name, ef_xref) for every /Filespec object in the document."""
    for xref in range(1, doc.xref_length()):
        try:
            obj = doc.xref_object(xref, compressed=False)
        except Exception:
            continue

        if "/Type /Filespec" not in obj:
            continue

        # Prefer /UF (Unicode filename) over /F (legacy)
        name = (
            _parse_pdf_literal_string(obj, "UF")
            or _parse_pdf_literal_string(obj, "F")
        )
        if not name:
            continue

        # The stream is at /EF << /F <ef_xref> R >>
        ef_block_match = re.search(r"/EF\s*<<([^>]+)>>", obj)
        if not ef_block_match:
            continue
        ef_xref = _extract_xref_ref(ef_block_match.group(1), "F")
        if ef_xref is None:
            continue

        yield name, ef_xref


def list_embedded_files(doc: fitz.Document) -> list[dict]:
    """Return metadata for every embedded filespec."""
    entries = []
    for i, (name, ef_xref) in enumerate(_iter_filespecs(doc)):
        try:
            size = len(doc.xref_stream(ef_xref))
        except Exception:
            size = 0
        entries.append({"index": i, "name": name, "ef_xref": ef_xref, "size": size})
    return entries


def extract_embedded_files(doc: fitz.Document, out_dir: Path) -> list[tuple[str, int]]:
    """Extract all embedded filespecs to out_dir.  Returns (filename, bytes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, (name, ef_xref) in enumerate(_iter_filespecs(doc)):
        # Sanitise name for filesystem
        safe = "".join(c if c not in r'\/:*?"<>|' else "_" for c in name)
        if not safe:
            safe = f"embedded_{i}"

        data: bytes = doc.xref_stream(ef_xref)
        dest = out_dir / safe
        if dest.exists():
            dest = out_dir / f"{dest.stem}_{i}{dest.suffix}"

        dest.write_bytes(data)
        results.append((safe, len(data)))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and unpack a PDF portfolio into individual files."
    )
    parser.add_argument("pdf", help="Path to the PDF file to inspect")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print embedded file names without extracting",
    )
    args = parser.parse_args()

    src = Path(args.pdf).expanduser().resolve()
    if not src.exists():
        print(f"ERROR: File not found: {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Opening  : {src.name}  ({src.stat().st_size / 1_048_576:.1f} MB)")

    doc = fitz.open(str(src))

    if not is_pdf_portfolio(doc):
        # Fall back to PyMuPDF's built-in embedded-file API just in case
        count = doc.embfile_count()
        if count:
            print(
                f"Not a PDF portfolio (no /Collection key) but contains "
                f"{count} embedded file(s).  Re-run with --list-only to inspect."
            )
        else:
            print("Not a PDF portfolio — no /Collection key and no embedded files.")
        doc.close()
        return

    print("Detected : PDF Portfolio  (/Collection key present)")

    entries = list_embedded_files(doc)
    total = len(entries)
    print(f"Contents : {total} embedded file(s)\n")

    if total == 0:
        print("  (no embedded files found despite /Collection key — portfolio may be corrupt)")
        doc.close()
        return

    name_width = min(max(len(e["name"]) for e in entries), 80)

    if args.list_only:
        for e in entries:
            print(f"  [{e['index']+1:>3}]  {e['name']:<{name_width}}  {e['size']/1024:>8.1f} KB")
        doc.close()
        return

    # Extract
    out_dir = src.parent / src.stem
    print(f"Output   : {out_dir}\n")

    extracted = extract_embedded_files(doc, out_dir)
    doc.close()

    w = min(max(len(n) for n, _ in extracted), 80)
    total_bytes = 0
    for name, nb in extracted:
        total_bytes += nb
        print(f"  {name:<{w}}  {nb / 1_048_576:>7.2f} MB")

    print(f"\n  {len(extracted)} file(s) extracted  →  {total_bytes / 1_048_576:.1f} MB total")
    print(f"  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
