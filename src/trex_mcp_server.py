"""
TREX Knowledge Graph — MCP Server

Tools:
  trex_query              — Answer a question using the knowledge base
  trex_retrieve           — Get raw passages (no LLM answer)
  trex_status             — Check collection index info in Qdrant
  info://youtranscripts_dom — Get DOM snapshot and accessibility (Hardcoded to reduce token usage) of youtubetranscripts.com URL
  trex_get_youtube_transcript — Fetch + save YouTube transcript
  trex_run_pipeline       — Run the full indexing pipeline

Usage:
  python trex_mcp_server.py          # stdio (for agent)
  python trex_mcp_server.py --http   # HTTP

Pain Point(s): 
  Re-adjustment of upper levels in TREX doesnt happen, if I add new information tot he same coolection. Meaning similar kind of information may or may not be
  re-aligned under the same summary node(s). 
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Ensure sibling modules are importable when spawned as subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import TREXConfig
from store.vector_store import QdrantManager
from store.graph_store import Neo4jManager
from retrieval.qdrant_retrieval import QdrantRetriever
from retrieval.entity_linker import QueryEntityLinker
from retrieval.graph_context import GraphContextBuilder
from ingestion.ingest import ingest_and_chunk
from store.embedder import embed_text_units
from extraction.extract import extract_and_store_graph
from raptor.raptor_tree import build_raptor_tree
from indexing.indexer import index_into_qdrant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")
logger = logging.getLogger("trex_mcp")

TRANSCRIPT_DIR = f"{PROJECT_ROOT}/ingestion/data"

logger.info("TREX MCP ready")

mcp = FastMCP("trex_mcp")

# ── Hardcoded DOM reference ────────────────────
# Agent can read this resource if the transcript tool breaks.
YOUTRANSCRIPTS_DOM = """\
URL: https://www.youtranscripts.com/transcript/{VIDEO_ID}/
Transcript text: .transcript-container p  (call .innerText)
Video title:     h1  (prefixed with "Transcript of")
URL input:       input[placeholder*="YouTube"]
Buttons:         "Generate Transcript", "Copy Transcript", "Download Transcript"
"""

# ── Helpers ──────────────────────────────────────────────────────────
def _extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|/v/|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
        return url
    raise ValueError(f"Cannot extract video ID from: {url}")

def _translate_to_english(text: str, config: TREXConfig) -> str:
    """Chunk text and translate each chunk Hindi → English via OpenAI."""
    
    client = OpenAI(base_url=config.openai_indexer_endpoint, api_key=config.openai_indexer_api_key)
    CHUNK_CHARS = 12000  # ~3K tokens input → plenty of headroom

    chunks = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
    translated = []

    for i, chunk in enumerate(chunks):
        logger.info("Translating chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
        resp = client.chat.completions.create(
            model=config.indexing_model,
            messages=[
                {"role": "system", "content": (
                    "You are a translator. Translate the following text to English. "
                    "Preserve all meaning, names, numbers, and structure. "
                    "Output ONLY the translation, nothing else. There may be English mixed with other languages so keep in mind while translating."
                )},
                {"role": "user", "content": chunk},
            ],
            temperature=0,
        )
        translated.append(resp.choices[0].message.content.strip())

    return "\n".join(translated)

async def _fetch_transcript_playwright(video_id: str) -> dict:
    """Use Playwright directly (not MCP) to scrape youtranscripts.com."""
    from playwright.async_api import async_playwright

    url = f"https://www.youtranscripts.com/transcript/{video_id}/"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True) # Doesnt open browser window, runs in background
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_selector(".transcript-container p", timeout=30000)
            return await page.evaluate("""() => {
                const p = document.querySelector('.transcript-container p');
                const h1 = document.querySelector('h1');
                const transcript = p ? p.innerText : '';
                const title = (h1?.innerText || '')
                    .replace(/^Transcript of\\s*["\u201C]?/, '')
                    .replace(/["\u201D]$/, '');
                return { transcript, title, char_count: transcript.length, success: transcript.length > 50 };
            }""")
        finally:
            await browser.close()


@mcp.resource("info://youtranscripts_dom")
def youtranscripts_dom() -> str:
    """Hardcoded DOM reference for youtranscripts.com. Read it before calling the get_youtube_transcript tool."""
    return YOUTRANSCRIPTS_DOM


@mcp.tool()
async def trex_retrieve(query: str, 
                        qdrant_collection: str = "test_collection", # no need for neo4j, the database name remains same.
                        top_k: int = 10, 
                        enable_graph: bool = False) -> str:
    """Retrieve raw passages from the knowledge base without generating an answer."""
    try:
        config = TREXConfig()
        qdrant = QdrantRetriever(config)
        neo4j = Neo4jManager(config) if enable_graph else None
        config.qdrant_collection = qdrant_collection

        results = qdrant.search(query, top_k=top_k)

        passages = []
        for r in results:
            passages.append({
                "text": r.get("text", ""),
                "title": r.get("title", "?"),
                "level": r.get("level", 0),
                "score": round(r.score, 4) if hasattr(r, "score") else None,
            })

        out: dict = {"passages": passages}

        if enable_graph and neo4j:
            linker = QueryEntityLinker(config, neo4j)
            linked = linker.extract_and_link(query)
            builder = GraphContextBuilder(config, neo4j)
            graph_text, _ = builder.build(linked)
            if graph_text:
                out["graph_context"] = graph_text

        return json.dumps(out, indent=2)

    except Exception as e:
        logger.exception("trex_retrieve failed")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def trex_status(collection_name: str | None, list_all: bool = False) -> str:
    """Return index status: collection name, point count, graph enabled."""
    try:
        config = TREXConfig()
        qdrant = QdrantManager(config)
        if list_all and not collection_name:
            # List all collections in Qdrant
            collections = qdrant.client.get_collections().collections
            collection_info = []
            for col in collections:
                info = qdrant.client.get_collection(col.name)
                collection_info.append({
                    "collection": col.name,
                    "points": info.points_count,
                    "status": info.status.value if info.status else "unknown",
                })
            return json.dumps({"collections": collection_info}, indent=2)
        
        config.qdrant_collection = collection_name
        info = qdrant.client.get_collection(config.qdrant_collection)
        return json.dumps({
            "collection": config.qdrant_collection,
            "points": info.points_count,
            "status": info.status.value if info.status else "unknown",
            "graph_enabled": config.enable_graph,
        }, indent=2)
    except ValueError as e:
        if "not found" in str(e):
            return json.dumps({"error": f"Collection '{collection_name}' not found"})
    except Exception as e:
        logger.exception("trex_status failed")
        return json.dumps({"error": str(e)})

@mcp.tool()
async def entity_neo4j_status() -> str:
    """Return Neo4j graph status: total node and relationship counts."""
    try:
        config = TREXConfig()
        config.enable_graph = True
        neo4j = Neo4jManager(config)
        if not config.enable_graph or not neo4j:
            return json.dumps({"error": "Graph is disabled in config"})

        def _query():
            driver = neo4j.driver
            with driver.session() as session:
                stats = session.run(
                    "CALL db.stats.retrieve('GRAPH COUNTS') YIELD data RETURN data"
                ).single()

                if stats:
                    return stats["data"]

                # Fallback if db.stats not available
                n = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                r = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
                labels = session.run("CALL db.labels()").data()
                rel_types = session.run("CALL db.relationshipTypes()").data()

                return {
                    "node_count": n,
                    "relationship_count": r,
                    "labels": [l["label"] for l in labels],
                    "relationship_types": [t["relationshipType"] for t in rel_types],
                }

        result = await asyncio.to_thread(_query)
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.exception("entity_neo4j_status failed")
        return json.dumps({"error": str(e)})

@mcp.tool()
async def trex_get_youtube_transcript(
    youtube_url: str,
    save_dir: str = TRANSCRIPT_DIR,
    download: bool = True,
    not_english_languages: bool = False
) -> str:
    """Fetch a YouTube transcript from youtranscripts.com and optionally save as .txt."""
    try:
        video_id = _extract_video_id(youtube_url)
        extraction = await _fetch_transcript_playwright(video_id)

        if not extraction or not extraction.get("success"):
            return json.dumps({"error": "Could not extract transcript", "video_id": video_id})

        transcript = extraction["transcript"]
        title = extraction.get("title", video_id)

        config = TREXConfig()
        if not_english_languages:
            transcript = _translate_to_english(transcript, config)

        file_path = None
        if download:
            os.makedirs(save_dir, exist_ok=True)
            safe = re.sub(r'[^\w\s-]', '', title)[:80].strip().replace(' ', '_')
            fname = f"transcript_{video_id}_{safe}.txt" if safe else f"transcript_{video_id}.txt"
            file_path = os.path.join(save_dir, fname)
            Path(file_path).write_text(transcript, encoding="utf-8")
            logger.info("Saved transcript to %s", file_path)

        result = {"video_id": video_id, "title": title, "char_count": len(transcript)}
        if file_path:
            result["file_path"] = file_path
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.exception("trex_get_youtube_transcript failed")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def trex_run_pipeline(
    sources: list[str],
    collection_name: str | None = None,
    enable_graph: bool = False,
    max_gleanings: int = 1,
) -> str:
    """Run the TREX indexing pipeline: ingest → embed → (graph) → RAPTOR → Qdrant."""
    # Run only entity graph?
    try:
        if collection_name is None:
            return json.dumps({"error": "collection_name is required"})
        
        # Validate sources exist
        missing = [s for s in sources if not os.path.exists(s)]
        if missing:
            return json.dumps({"error": f"Files not found: {missing}"})

        # Build config for this run
        run_config = TREXConfig(
            qdrant_collection=collection_name,
            enable_graph=enable_graph,
        )
        qdrant = QdrantManager(run_config)
        info = qdrant.client.get_collection(collection_name)

        def _run():
            if info.points_count > 0:
                logger.warning("Collection '%s' already has %d points. New data will be added to it.", collection_name, info.points_count)
            
            text_units = ingest_and_chunk(sources=sources, config=run_config)
            logger.info("Step 1: %d chunks", len(text_units))

            text_units = embed_text_units(text_units, run_config)
            logger.info("Step 2: embedded")

            neo4j_mgr = None
            if run_config.enable_graph:
                neo4j_mgr = Neo4jManager(run_config)
                text_units = extract_and_store_graph(text_units, run_config, neo4j_mgr, max_gleanings=max_gleanings)
                logger.info("Step 3: graph extracted")

            all_nodes = build_raptor_tree(text_units, run_config)
            logger.info("Step 4: %d nodes", len(all_nodes))

            levels = {}
            for n in all_nodes:
                levels[n.level] = levels.get(n.level, 0) + 1

            index_into_qdrant(all_nodes, run_config)
            logger.info("Step 5: indexed to '%s'", run_config.qdrant_collection)

            if neo4j_mgr:
                neo4j_mgr.close()

            return {
                "sources": sources,
                "chunks": len(text_units),
                "total_nodes": len(all_nodes),
                "levels": levels,
                "graph_enabled": run_config.enable_graph,
                "collection": run_config.qdrant_collection,
            }

        summary = await asyncio.to_thread(_run)
        return json.dumps(summary, indent=2)

    except ValueError as e:
        if "not found" in str(e):
            return json.dumps({"error": f"Collection '{collection_name}' not found"})
    
    except Exception as e:
        logger.exception("trex_run_pipeline failed")
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