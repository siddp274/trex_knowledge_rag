"""
Step 1: Document Ingestion & Chunking

This is the entry point for the indexing pipeline. It:
1. Loads documents from multiple sources (PDF, HTML, CSV, TXT, MD, URLs)
2. Concatenates multi-page documents into a single text per source
3. Optionally prepends metadata (title, date) to each chunk
4. Splits into token-counted chunks using GraphRAG's algorithm
5. Returns TextUnit objects ready for Step 2 (embedding)

Usage:
    from ingestion.ingest import ingest_and_chunk
    from config import TREXConfig
    
    config = TREXConfig()
    text_units = ingest_and_chunk(
        sources=["report.pdf", "https://example.com/article"],
        config=config,
    )
"""
import logging
from typing import Any
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TREXConfig
from ingestion.hashing import gen_sha512_hash
from ingestion.loader import load_single_source
from ingestion.chunker import TokenChunker, ChunkingConfig, add_metadata_transformer
from ingestion.text_unit import TextUnit

logger = logging.getLogger(__name__)


def ingest_and_chunk(
    sources: list[str],
    config: TREXConfig,
) -> list[TextUnit]:
    """
    Full Step 1: Load → Concatenate → Chunk → Return TextUnits.
    
    Args:
        sources: List of file paths or URLs to ingest.
        config: Pipeline configuration.
    
    Returns:
        List of TextUnit objects (level=0 leaf nodes) with:
        - Deterministic IDs (SHA-512 of text)
        - Exact token counts
        - Metadata-prepended text (if configured)
        - Document provenance (source, title, document_id)
        - Empty slots for embedding, entity_ids, etc. (filled in later steps)
    """
    chunking_config = ChunkingConfig(
        size=config.chunk_size,
        overlap=config.chunk_overlap,
        prepend_metadata=["title"],  # Prepend document title to every chunk
    )
    chunker = TokenChunker(chunking_config)

    all_text_units: list[TextUnit] = []

    for source in sources:
        logger.info(f"[Ingest] Loading: {source}")

        try:
            documents = load_single_source(source)
        except Exception as e:
            logger.error(f"[Ingest] FAILED to load {source}: {e}")
            continue

        if not documents:
            logger.warning(f"[Ingest] No text extracted from {source}")
            continue

        # Concatenate all pages/sections from this sources like PDFs and CSVs
        full_text = "\n\n".join(doc["text"] for doc in documents)
        
        # Use the first document's metadata as representative
        doc_meta = documents[0]
        document_id = gen_sha512_hash({"source": source, "text": full_text[:500]}, ["source", "text"])
        title = doc_meta.get("title", source)

        # --- Build metadata transformer (GraphRAG pattern) ---
        # Prepends configured metadata fields to each chunk.
        # e.g., "title: Annual Report 2024.\n<chunk text>"
        transform = None
        if chunking_config.prepend_metadata:
            metadata = {}
            for field in chunking_config.prepend_metadata:
                value = doc_meta.get(field)
                if value:
                    metadata[field] = value
            if metadata:
                transform = add_metadata_transformer(metadata)

        chunks = chunker.chunk(full_text, transform=transform)

        logger.info(f"[Ingest] {source} → {len(chunks)} chunks")

        # --- Create TextUnit objects ---
        for chunk_data in chunks:
            text_unit = TextUnit(
                id=gen_sha512_hash({"text": chunk_data["text"]}, ["text"]),
                text=chunk_data["text"],
                original_text=chunk_data["original_text"],
                document_id=document_id,
                source=source,
                title=title,
                n_tokens=chunk_data["n_tokens"],
                level=0,          # leaf node
                parent_id=None,
                children_ids=[],
                entity_ids=None,  # filled in Step 3
                relationship_ids=None,  # filled in Step 3
                embedding=None,   # filled in Step 2
            )
            all_text_units.append(text_unit)

    # --- Summary stats ---
    total_tokens = sum(tu.n_tokens for tu in all_text_units)
    unique_sources = len(set(tu.source for tu in all_text_units))
    logger.info(f"[Ingest] Complete: {len(all_text_units)} text units from {unique_sources} sources")
    logger.info(f"[Ingest] Total tokens: {total_tokens:,}")
    logger.info(f"[Ingest] Avg tokens per chunk: {total_tokens // max(len(all_text_units), 1)}")
    logger.info(f"[Ingest] Estimated embedding cost: ${total_tokens * 0.02 / 1_000_000:.4f}")

    return all_text_units

    