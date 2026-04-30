"""
Standalone script to reindex a single document using the current config + code.

Usage:
    python tools/reindex_one.py <file_path> [--vision-mode threshold|landscape]

The optional --vision-mode flag overrides the VISION_MODE env/config value for
this run only.  Useful for document categories whose orientation differs from
the global default (e.g. portrait-mode magazines when the global mode is
'landscape').
"""
import argparse
import asyncio
import io
import logging
import sys
from pathlib import Path

# Ensure stdout can handle Unicode (Windows cp1252 default fails on e.g. ✅)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Make sure we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stderr,
)

async def main():
    ap = argparse.ArgumentParser(description="Force-reindex a single document.")
    ap.add_argument("file_path", help="Path to the document to reindex")
    ap.add_argument(
        "--vision-mode",
        choices=["threshold", "landscape"],
        default=None,
        help="Override VISION_MODE for this run (default: use config/env value)",
    )
    args = ap.parse_args()

    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    from src.config import get_config
    from src.parser import DocumentParser
    from src.llm import LocalLLM, DocumentProcessor
    from src.indexer import DocumentIndexer

    config = get_config()
    vision_mode = args.vision_mode or config.vision_mode
    print(f"Config: vision_enhancement={config.enable_vision_enhancement}, "
          f"mode={vision_mode}{' (override)' if args.vision_mode else ''}, "
          f"lancedb={config.lancedb_path}", file=sys.stderr)

    # Parser
    parser = DocumentParser(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        use_structured=config.use_structured_parsing,
        max_chunk_tokens=config.max_chunk_tokens,
        enable_ocr=config.enable_ocr,
        ocr_language=config.ocr_language,
        enable_vision_summary=config.enable_vision_summary,
        vision_model=config.vision_model,
        enable_vision_enhancement=config.enable_vision_enhancement,
        vision_word_threshold=config.vision_word_threshold,
        vision_mode=vision_mode,
    )

    # LLM + processor
    llm = LocalLLM(
        model=config.llm_model,
        base_url=config.ollama_base_url,
    )
    await llm.initialize()
    processor = DocumentProcessor(llm)

    # Indexer
    indexer = DocumentIndexer(
        db_path=config.lancedb_path,
        embedding_model=config.embedding_model,
    )
    await indexer.initialize()

    print(f"\n[1/3] Parsing {file_path.name} …", file=sys.stderr)
    doc_data = parser.parse_file(file_path)
    print(f"      {doc_data['num_chunks']} chunks, {doc_data['total_chars']:,} chars", file=sys.stderr)

    print("[2/3] Processing with LLM …", file=sys.stderr)
    doc_data = await processor.process_document(doc_data)

    print("[3/3] Indexing (force) …", file=sys.stderr)
    # Force reindex by removing the old entry first so the hash check doesn't skip it.
    await indexer.remove_document(str(file_path))
    success = await indexer.index_document(doc_data)
    print(f"      Result: {'indexed' if success else 'ERROR — index_document returned False'}", file=sys.stderr)

    # Print chunk summary for vision_description chunks
    vision_chunks = [c for c in doc_data["chunks"] if c.get("element_type") == "vision_description"]
    if vision_chunks:
        print(f"\n--- Vision description chunks ({len(vision_chunks)} total) ---")
        for c in vision_chunks:
            text = c["text"]
            print(f"\n[chunk_id={c['chunk_id']} | page={c['page_or_sheet']} | "
                  f"section={c['section_path']} | chars={c['char_count']}]")
            print(text)
    else:
        print("\nNo vision_description chunks produced.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
