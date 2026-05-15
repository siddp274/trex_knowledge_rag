"""
TREX Knowledge Graph — MCP Server

Tools:
  trex_retrieve   — Retrieve raw passages from Qdrant (no LLM answer)
  trex_status     — Check collection index info in Qdrant
  trex_run_pipeline — Run the full indexing pipeline

Pain Point(s):
  Re-adjustment of upper levels in TREX doesn't happen when new information is added to
  the same collection. Similar content may not be re-aligned under the same summary node(s).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import TREXConfig
from store.vector_store import QdrantManager
from retrieval.context_assembler import ContextAssembler, QueryPipeline
from ingestion.ingest import ingest_and_chunk
from store.embedder import embed_text_units
from raptor.raptor_tree import build_raptor_tree
from indexing.indexer import index_into_qdrant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")
logger = logging.getLogger("trex_mcp")

logger.info("TREX MCP ready")

mcp = FastMCP("trex_mcp")


@mcp.tool()
async def trex_retrieve(query: str, qdrant_collection: str, top_k: int = 10) -> str:
    """Retrieve raw passages from the knowledge base without generating an answer."""
    try:
        config = TREXConfig()
        if not qdrant_collection:
            return json.dumps({"error": "qdrant_collection name is required"})

        config.qdrant_collection = qdrant_collection
        qdrant = QueryPipeline(config)
        results = qdrant._retrieve_context(query, top_k=top_k, max_context_tokens=1650)
        logger.info("[trex_retrieve] query=%r collection=%s top_k=%d → %d results", query, qdrant_collection, top_k, len(results))
        return json.dumps({"passages": results}, indent=2)

    except Exception as e:
        logger.exception("[trex_retrieve] failed")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def trex_status(collection_name: str | None, list_all: bool = False) -> str:
    """Return index status: collection name and point count."""
    try:
        config = TREXConfig()
        qdrant = QdrantManager(config)

        if list_all and not collection_name:
            collection_info = qdrant.collection_list()
            logger.info("[trex_status] Listing all collections")
            return json.dumps({"collections": collection_info}, indent=2)

        config.qdrant_collection = collection_name
        info = qdrant._ensure_collection()
        logger.info("[trex_status] collection=%s info=%s", collection_name, info)
        return json.dumps(info, indent=2)

    except ValueError as e:
        if "not found" in str(e):
            return json.dumps({"error": f"Collection '{collection_name}' not found"})
    except Exception as e:
        logger.exception("[trex_status] failed")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def trex_run_pipeline(sources: list[str], collection_name: str | None = None) -> str:
    """Run the TREX indexing pipeline: ingest → embed → RAPTOR → Qdrant.
    """
    try:
        if collection_name is None:
            return json.dumps({"error": "collection_name is required"})

        missing = [s for s in sources if not os.path.exists(s)]
        if missing:
            return json.dumps({"error": f"Files not found: {missing}"})

        run_config = TREXConfig(qdrant_collection=collection_name)
        qdrant = QdrantManager(run_config)
        info = qdrant.client.get_collection(collection_name)

        def _run():
            if info.points_count > 0:
                logger.warning("[trex_run_pipeline] Collection '%s' already has %d points. New data will be added.", collection_name, info.points_count)

            text_units = ingest_and_chunk(sources=sources, config=run_config)
            logger.info("[trex_run_pipeline] Step 1 — ingested %d chunks", len(text_units))

            text_units = embed_text_units(text_units, run_config)
            logger.info("[trex_run_pipeline] Step 2 — embeddings done")

            all_nodes = build_raptor_tree(text_units, run_config)
            logger.info("[trex_run_pipeline] Step 3 — RAPTOR tree: %d nodes", len(all_nodes))

            levels = {}
            for n in all_nodes:
                levels[n.level] = levels.get(n.level, 0) + 1

            index_into_qdrant(all_nodes, run_config)
            logger.info("[trex_run_pipeline] Step 4 — indexed to '%s'", run_config.qdrant_collection)

            return {
                "sources": sources,
                "chunks": len(text_units),
                "total_nodes": len(all_nodes),
                "levels": levels,
                "collection": run_config.qdrant_collection,
            }

        summary = await asyncio.to_thread(_run)
        return json.dumps(summary, indent=2)

    except ValueError as e:
        if "not found" in str(e):
            return json.dumps({"error": f"Collection '{collection_name}' not found"})
    except Exception as e:
        logger.exception("[trex_run_pipeline] failed")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="streamable_http", port=args.port)
    else:
        mcp.run()
