import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TREXConfig
from ingestion.ingest import ingest_and_chunk
from store.embedder import embed_text_units
from extraction.extract import extract_and_store_graph
from raptor.raptor_tree import build_raptor_tree
from indexing.indexer import index_into_qdrant
from store.vector_store import QdrantManager
from store.graph_store import Neo4jManager
from retrieval.pipeline import QueryPipeline

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


config = TREXConfig(
    qdrant_collection="trex_paper_collection",
    neo4j_database="neo4j",
    enable_graph=False,
)

# ============================================================
# INDEXING (run once per corpus)
# ============================================================
qdrant = QdrantManager(config)
neo4j = None

# Step 1: Ingest & Chunk
text_units = ingest_and_chunk(
    sources=["/Users/siddp278/Desktop/projects/graphRAG/papers/2404.16130v2.pdf",
             "/Users/siddp278/Desktop/projects/graphRAG/papers/2503.02922v1.pdf",
             "/Users/siddp278/Desktop/projects/graphRAG/papers/8997_RAPTOR_Recursive_Abstract.pdf",
             "/Users/siddp278/Desktop/projects/graphRAG/papers/cormacksigir09-rrf.pdf"],
    config=config,
)

# Step 2: Embed
text_units = embed_text_units(text_units, config)

if config.enable_graph:
# Step 3: Extract entities/relationships → Neo4j
    neo4j = Neo4jManager(config)
    text_units = extract_and_store_graph(text_units, config, neo4j, max_gleanings=1)
    neo4j.close()

# Step 4: Build RAPTOR tree
all_nodes = build_raptor_tree(text_units, config)

# Step 5: Index into Qdrant
index_into_qdrant(all_nodes, config)

# ============================================================
# QUERYING (run per user question)
# ============================================================
pipeline = QueryPipeline(config, neo4j)

# OLTP-style (factual) — hits Level 0 chunks via BM25
result = pipeline.query("What was the performance upgrade of TREX over GraphRAG in terms of token count?")
print(result["answer"])

# OLAP-style (thematic) — hits Level 1-2 summaries via dense search
result = pipeline.query("What are the key strategic themes across all documents?")
print(result["answer"])

# Streaming
for token in pipeline.query_stream("How are the papers connected to each other?"):
    print(token, end="", flush=True)
