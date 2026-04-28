"""LanceDB indexing operations for document storage and retrieval."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import lancedb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# New structured fields added to the chunks table.
# Stored here so both _ensure_tables and _migrate_chunks_schema reference
# the same list.
# ---------------------------------------------------------------------------
_STRUCTURED_CHUNK_FIELDS = [
    ("element_type",    pa.string()),   # paragraph | heading_1 | table | sheet | figure …
    ("section_path",    pa.string()),   # breadcrumb e.g. "Chapter 1 > Section 2"
    ("page_or_sheet",   pa.string()),   # page number (str) or sheet name
    ("embedding_text",  pa.string()),   # enriched text actually sent to the embedding model
    ("html_or_markdown", pa.string()),  # markdown table or "" for non-table chunks
    ("extracted_by",    pa.string()),   # "docling" | "easyocr" | "" (PDF chunks only)
]


class DocumentIndexer:
    """Manage document indexing with LanceDB."""

    def __init__(self, db_path: Path, embedding_model: str):
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.embedding_model: Optional[SentenceTransformer] = None
        self.db = None
        self.catalog_table = None
        self.chunks_table = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        # Set to True if the live chunks table is missing the new columns
        # (happens when an existing DB was created before this upgrade).
        self._legacy_chunks_schema: bool = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self):
        """Initialize the database and embedding model."""
        logger.info("Loading embedding model: %s", self.embedding_model_name)
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        logger.info("Initializing LanceDB at: %s", self.db_path)
        self.db = await asyncio.get_event_loop().run_in_executor(
            self.executor, lancedb.connect, str(self.db_path)
        )

        await self._ensure_tables()

    async def _ensure_tables(self):
        """Create or open the catalog and chunks tables."""
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        catalog_schema = pa.schema([
            pa.field("file_path",    pa.string()),
            pa.field("file_name",    pa.string()),
            pa.field("file_hash",    pa.string()),
            pa.field("file_size",    pa.int64()),
            pa.field("file_type",    pa.string()),
            pa.field("modified_time", pa.string()),
            pa.field("indexed_time", pa.string()),
            pa.field("summary",      pa.string()),
            pa.field("keywords",     pa.string()),
            pa.field("total_chunks", pa.int64()),
            pa.field("total_chars",  pa.int64()),
            pa.field("total_tokens", pa.int64()),
            pa.field("embedding",    pa.list_(pa.float32(), embedding_dim)),
        ])

        # Full new schema for chunks (used only when creating a brand-new table)
        chunks_schema = pa.schema(
            [
                pa.field("file_path",  pa.string()),
                pa.field("file_hash",  pa.string()),
                pa.field("chunk_id",   pa.int64()),
                pa.field("chunk_text", pa.string()),
                pa.field("start_pos",  pa.int64()),
                pa.field("end_pos",    pa.int64()),
                pa.field("char_count", pa.int64()),
                pa.field("token_count", pa.int64()),
            ]
            + [pa.field(name, dtype) for name, dtype in _STRUCTURED_CHUNK_FIELDS]
            + [pa.field("embedding", pa.list_(pa.float32(), embedding_dim))]
        )

        existing_tables = await asyncio.get_event_loop().run_in_executor(
            self.executor, self.db.table_names
        )

        # --- catalog table ---
        if "catalog" not in existing_tables:
            logger.info("Creating catalog table")
            dummy_emb = np.zeros(embedding_dim, dtype=np.float32).tolist()
            init_data = pa.table(
                {
                    "file_path": ["dummy"], "file_name": ["dummy"],
                    "file_hash": ["dummy"], "file_size": [0],
                    "file_type": ["dummy"], "modified_time": ["2024-01-01"],
                    "indexed_time": ["2024-01-01"], "summary": ["dummy"],
                    "keywords": ["[]"], "total_chunks": [0],
                    "total_chars": [0], "total_tokens": [0],
                    "embedding": [dummy_emb],
                },
                schema=catalog_schema,
            )
            self.catalog_table = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.db.create_table, "catalog", init_data
            )
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.catalog_table.delete("file_path = 'dummy'"),
            )
        else:
            self.catalog_table = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.db.open_table, "catalog"
            )

        # --- chunks table ---
        if "chunks" not in existing_tables:
            logger.info("Creating chunks table (new schema)")
            dummy_emb = np.zeros(embedding_dim, dtype=np.float32).tolist()
            init_data = pa.table(
                {
                    "file_path": ["dummy"], "file_hash": ["dummy"],
                    "chunk_id": [0], "chunk_text": ["dummy"],
                    "start_pos": [0], "end_pos": [0],
                    "char_count": [0], "token_count": [0],
                    **{name: [""] for name, _ in _STRUCTURED_CHUNK_FIELDS},
                    "embedding": [dummy_emb],
                },
                schema=chunks_schema,
            )
            self.chunks_table = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.db.create_table, "chunks", init_data
            )
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.chunks_table.delete("file_path = 'dummy'"),
            )
        else:
            self.chunks_table = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.db.open_table, "chunks"
            )
            # Attempt to add new columns to an existing (pre-upgrade) table
            await self._migrate_chunks_schema()

    async def _migrate_chunks_schema(self):
        """Add structured fields to an existing chunks table if they are missing."""
        try:
            existing_cols = {f.name for f in self.chunks_table.schema}
            missing = {
                name: "cast('' as string)"
                for name, _ in _STRUCTURED_CHUNK_FIELDS
                if name not in existing_cols
            }
            if not missing:
                return  # Already up to date

            logger.info(
                "Migrating chunks table — adding columns: %s", list(missing.keys())
            )
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.chunks_table.add_columns(missing),
            )
            logger.info("Chunks table migration complete")

        except Exception as exc:
            logger.warning(
                "Could not migrate chunks table to new schema: %s. "
                "Structured metadata fields will not be stored for documents "
                "indexed before this upgrade. Re-index to populate them.",
                exc,
            )
            self._legacy_chunks_schema = True

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_document(self, doc_data: Dict[str, Any]) -> bool:
        """Index a document and its chunks.  Returns True if newly indexed."""
        try:
            metadata   = doc_data["metadata"]
            file_path  = metadata["file_path"]
            file_hash  = metadata["file_hash"]
            file_name  = metadata["file_name"]
            num_chunks = doc_data["num_chunks"]

            logger.info("    -> Checking if document is already indexed...")
            if await self.is_document_indexed(file_path, file_hash):
                logger.info("    -> Already indexed (same hash): %s", file_name)
                return False

            logger.info("    -> Removing old version if exists...")
            await self.remove_document(file_path)

            logger.info("    -> Generating document-level embedding...")
            embedding_text = doc_data.get("embedding_text") or doc_data.get("summary", "")
            if not embedding_text and doc_data.get("chunks"):
                embedding_text = doc_data["chunks"][0]["text"]
            doc_embedding = await self._generate_embedding(embedding_text)
            logger.info("    -> Document embedding: %d dims", len(doc_embedding))

            # --- catalog entry ---
            catalog_entry = {
                "file_path":    file_path,
                "file_name":    metadata["file_name"],
                "file_hash":    file_hash,
                "file_size":    metadata["file_size"],
                "file_type":    metadata["file_type"],
                "modified_time": metadata["modified_time"],
                "indexed_time": datetime.now().isoformat(),
                "summary":      doc_data.get("summary", ""),
                "keywords":     json.dumps(doc_data.get("keywords", [])),
                "total_chunks": num_chunks,
                "total_chars":  doc_data["total_chars"],
                "total_tokens": doc_data["total_tokens"],
                "embedding":    doc_embedding.astype(np.float32).tolist(),
            }
            logger.info("    -> Adding to catalog...")
            catalog_df = pd.DataFrame([catalog_entry])
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self.catalog_table.add, catalog_df
            )

            # --- chunk entries ---
            if doc_data["chunks"]:
                logger.info("    -> Embedding %d chunks...", num_chunks)
                chunk_entries: List[Dict] = []

                for i, chunk in enumerate(doc_data["chunks"]):
                    if (i + 1) % 5 == 0 or i == num_chunks - 1:
                        logger.info(
                            "    -> Embedding chunk %d/%d", i + 1, num_chunks
                        )

                    # Use enriched embedding_text when available; fall back to raw text.
                    embed_text = chunk.get("embedding_text") or chunk.get("text", "")
                    chunk_embedding = await self._generate_embedding(embed_text)

                    entry: Dict[str, Any] = {
                        "file_path":   file_path,
                        "file_hash":   file_hash,
                        "chunk_id":    chunk.get("chunk_id", i),
                        "chunk_text":  chunk.get("text", ""),
                        "start_pos":   chunk.get("start_pos", -1),
                        "end_pos":     chunk.get("end_pos", -1),
                        "char_count":  chunk.get("char_count", len(chunk.get("text", ""))),
                        "token_count": chunk.get("token_count", 0),
                        "embedding":   chunk_embedding.astype(np.float32).tolist(),
                    }

                    # Add structured fields only when the schema supports them
                    if not self._legacy_chunks_schema:
                        entry["element_type"]    = chunk.get("element_type", "paragraph")
                        entry["section_path"]    = chunk.get("section_path", "")
                        entry["page_or_sheet"]   = chunk.get("page_or_sheet") or ""
                        entry["embedding_text"]  = chunk.get("embedding_text", "")
                        entry["html_or_markdown"] = chunk.get("html_or_markdown", "")
                        entry["extracted_by"]    = chunk.get("extracted_by", "")

                    chunk_entries.append(entry)

                logger.info("    -> Storing %d chunks...", num_chunks)
                chunks_df = pd.DataFrame(chunk_entries)
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.chunks_table.add, chunks_df
                )
                logger.info("    -> All %d chunks stored", num_chunks)
            else:
                logger.warning("    -> No chunks to index for %s", file_name)

            # --- log chunk type breakdown ---
            type_counts: Dict[str, int] = {}
            for chunk in doc_data["chunks"]:
                etype = chunk.get("element_type", "paragraph")
                type_counts[etype] = type_counts.get(etype, 0) + 1
            if type_counts:
                breakdown = ", ".join(
                    f"{k}={v}" for k, v in sorted(type_counts.items())
                )
                logger.info(
                    "Successfully indexed: %s | %d chunks [%s]",
                    file_name, num_chunks, breakdown,
                )
            else:
                logger.info(
                    "Successfully indexed: %s (%d chunks)", file_name, num_chunks
                )
            return True

        except Exception as exc:
            logger.error(
                "Error indexing %s: %s",
                doc_data.get("metadata", {}).get("file_path", "unknown"),
                exc,
            )
            raise

    # ------------------------------------------------------------------
    # Queries (unchanged behaviour)
    # ------------------------------------------------------------------

    async def is_document_indexed(self, file_path: str, file_hash: str) -> bool:
        """Return True if document is already indexed with the same hash."""
        try:
            query = f"file_path = '{file_path}' AND file_hash = '{file_hash}'"
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.catalog_table.search().where(query).limit(1).to_pandas(),
            )
            return len(results) > 0
        except Exception:
            return False

    async def get_indexed_files(self) -> Dict[str, Dict[str, str]]:
        """Return dict of file_path -> {file_hash, modified_time} for all indexed files."""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.catalog_table.to_pandas()[
                    ["file_path", "file_hash", "modified_time"]
                ],
            )
            return {
                row["file_path"]: {
                    "file_hash": row["file_hash"],
                    "modified_time": row["modified_time"],
                }
                for _, row in results.iterrows()
            }
        except Exception:
            return {}

    async def remove_document(self, file_path: str):
        """Remove a document and its chunks from the index."""
        try:
            info = await self.get_document_info(file_path)
            if not info:
                return False
            query = f"file_path = '{file_path}'"
            await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.catalog_table.delete(query)
            )
            await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.chunks_table.delete(query)
            )
            logger.info("Removed from index: %s", file_path)
            return True
        except Exception as exc:
            logger.error("Error removing %s: %s", file_path, exc)
            return False

    async def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Semantic search over document-level embeddings."""
        q_emb = (await self._generate_embedding(query)).astype(np.float32)
        results = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.catalog_table.search(q_emb, vector_column_name="embedding")
            .limit(limit).to_pandas(),
        )
        docs = []
        for _, row in results.iterrows():
            doc = row.to_dict()
            doc["keywords"] = json.loads(doc.get("keywords", "[]"))
            doc.pop("embedding", None)
            docs.append(doc)
        return docs

    async def search_chunks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Semantic search over chunk-level embeddings."""
        q_emb = (await self._generate_embedding(query)).astype(np.float32)
        results = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.chunks_table.search(q_emb, vector_column_name="embedding")
            .limit(limit).to_pandas(),
        )
        chunks = []
        for _, row in results.iterrows():
            chunk = row.to_dict()
            chunk.pop("embedding", None)
            chunks.append(chunk)
        return chunks

    async def get_catalog(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Return paginated list of all indexed documents."""
        results = await asyncio.get_event_loop().run_in_executor(
            self.executor, lambda: self.catalog_table.to_pandas()
        )
        results = results.sort_values("indexed_time", ascending=False).iloc[skip: skip + limit]
        docs = []
        for _, row in results.iterrows():
            doc = row.to_dict()
            doc["keywords"] = json.loads(doc.get("keywords", "[]"))
            doc.pop("embedding", None)
            docs.append(doc)
        return docs

    async def get_document_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Return full metadata for a single document."""
        query = f"file_path = '{file_path}'"
        results = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.catalog_table.search().where(query).limit(1).to_pandas(),
        )
        if len(results) == 0:
            return None
        doc = results.iloc[0].to_dict()
        doc["keywords"] = json.loads(doc.get("keywords", "[]"))
        doc.pop("embedding", None)
        chunk_results = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.chunks_table.search().where(query).to_pandas(),
        )
        doc["actual_chunks"] = len(chunk_results)
        return doc

    async def get_stats(self) -> Dict[str, Any]:
        """Return indexing statistics."""
        catalog_df = await asyncio.get_event_loop().run_in_executor(
            self.executor, lambda: self.catalog_table.to_pandas()
        )
        chunks_df = await asyncio.get_event_loop().run_in_executor(
            self.executor, lambda: self.chunks_table.to_pandas()
        )

        stats: Dict[str, Any] = {
            "total_documents": len(catalog_df),
            "total_chunks":    len(chunks_df),
            "total_size_bytes": int(catalog_df["file_size"].sum()) if len(catalog_df) > 0 else 0,
            "total_chars":     int(catalog_df["total_chars"].sum()) if len(catalog_df) > 0 else 0,
            "total_tokens":    int(catalog_df["total_tokens"].sum()) if len(catalog_df) > 0 else 0,
            "file_types":      catalog_df["file_type"].value_counts().to_dict() if len(catalog_df) > 0 else {},
            "db_path":         str(self.db_path),
        }

        # If the schema supports it, show chunk type breakdown
        if not self._legacy_chunks_schema and "element_type" in chunks_df.columns:
            stats["chunk_types"] = chunks_df["element_type"].value_counts().to_dict()

        return stats

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text string."""
        if not text:
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self.embedding_model.encode, text
        )

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in a single batch call."""
        if not texts:
            return []
        embeddings = await asyncio.get_event_loop().run_in_executor(
            self.executor, self.embedding_model.encode, texts
        )
        return list(embeddings)

    async def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)
