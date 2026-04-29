#!/usr/bin/env python3
"""Re-index a single document into the live LanceDB index.

Usage:
    uv run python scripts/reindex_file.py "path/to/document.pdf"

Always removes the existing chunks for the document first, then re-parses
and re-indexes using the full pipeline (parse → LLM summary → embeddings).
The file hash check is bypassed so the document is always refreshed.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow `src` imports when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import get_config
from src.parser import DocumentParser
from src.llm import LocalLLM, DocumentProcessor
from src.indexer import DocumentIndexer


async def reindex(file_path: Path) -> None:
    config = get_config()

    print(f"Live index : {config.lancedb_path}")
    print(f"File       : {file_path}")
    print()

    # Initialise components
    parser = DocumentParser(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        use_structured=config.use_structured_parsing,
        max_chunk_tokens=config.max_chunk_tokens,
        enable_ocr=config.enable_ocr,
        ocr_language=config.ocr_language,
        enable_vision_summary=config.enable_vision_summary,
        vision_model=config.vision_model,
    )

    llm = LocalLLM(model=config.llm_model, base_url=config.ollama_base_url)
    await llm.initialize()

    processor = DocumentProcessor(llm)

    indexer = DocumentIndexer(
        db_path=config.lancedb_path,
        embedding_model=config.embedding_model,
    )
    await indexer.initialize()

    # Force-remove any existing entry so the hash check in index_document
    # cannot short-circuit the write even when the file is unchanged on disk.
    print("[1/4] Removing existing index entry (if any)...")
    await indexer.remove_document(str(file_path))

    # Parse
    print("[2/4] Parsing document...")
    doc_data = parser.parse_file(file_path)
    print(f"      {doc_data['total_chars']:,} chars · {doc_data['num_chunks']} chunks · {doc_data['total_tokens']:,} tokens")

    # LLM summary / keywords
    print(f"[3/4] Processing with LLM ({config.llm_model})...")
    doc_data = await processor.process_document(doc_data)
    print(f"      Summary: {len(doc_data.get('summary', ''))} chars · Keywords: {doc_data.get('keywords', [])[:5]}")

    # Embed and write to LanceDB
    print("[4/4] Generating embeddings and writing to LanceDB...")
    success = await indexer.index_document(doc_data)

    print()
    if success:
        # Fetch the fresh catalog entry — filter in pandas to avoid SQL escaping issues
        import lancedb
        db = lancedb.connect(config.lancedb_path)
        catalog_table = db.open_table("catalog")
        all_rows = catalog_table.to_pandas()
        rows = all_rows[all_rows["file_path"] == str(file_path)]

        if not rows.empty:
            row = rows.iloc[0]
            print("=" * 60)
            print(f"  indexed_time : {row['indexed_time']}")
            print(f"  chunk count  : {int(row['total_chunks'])}")
            print(f"  total chars  : {int(row['total_chars']):,}")
            print(f"  file_hash    : {row['file_hash'][:16]}...")
            print("=" * 60)
        else:
            print("WARNING: document not found in catalog after indexing.")
    else:
        print("WARNING: index_document returned False — document may not have been written.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-index a single document into the live LanceDB index."
    )
    parser.add_argument("file", help="Absolute or relative path to the document to re-index")
    args = parser.parse_args()

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(reindex(file_path))


if __name__ == "__main__":
    main()
