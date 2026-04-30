#!/usr/bin/env python3
"""
Standalone test for vision enhancement on a single PDF.

Runs the vision enhancer against a PDF file (vision enabled), prints the
first generated vision-description chunk, and reports how many pages were
flagged as visually rich.

Usage:
    uv run python scripts/vision_test.py <path/to/file.pdf> [--threshold N]

The ANTHROPIC_API_KEY env var must be set (or present in .env).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.vision_enhancer import enhance_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test vision enhancement on a single PDF."
    )
    parser.add_argument("file", help="Path to the PDF file")
    parser.add_argument(
        "--mode",
        choices=["threshold", "landscape"],
        default="threshold",
        help="Invocation rule: 'threshold' (word count) or 'landscape' (orientation)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Word-count threshold (only used with --mode threshold, default: 50)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rasterisation DPI (default: 150)",
    )
    args = parser.parse_args()

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"File      : {file_path.name}")
    print(f"Mode      : {args.mode}")
    if args.mode == "threshold":
        print(f"Threshold : {args.threshold} words")
    print(f"DPI       : {args.dpi}")
    print()

    vision_elements = enhance_pdf(
        file_path,
        word_threshold=args.threshold,
        dpi=args.dpi,
        mode=args.mode,
    )

    print()
    print(f"Vision-description chunks generated: {len(vision_elements)}")

    if not vision_elements:
        print("No visually-rich pages found (or API call failed). Done.")
        return

    print()
    print("=" * 70)
    print("FIRST VISION-DESCRIPTION CHUNK")
    print("=" * 70)
    first = vision_elements[0]
    print(f"page_or_sheet : {first['page_or_sheet']}")
    print(f"section_path  : {first['section_path']}")
    print(f"element_type  : {first['element_type']}")
    print(f"extracted_by  : {first['extracted_by']}")
    print(f"text length   : {len(first['text'])} chars")
    print()
    print(first["text"])
    print("=" * 70)


if __name__ == "__main__":
    main()
