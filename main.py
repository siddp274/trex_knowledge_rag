import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import TREXConfig
from ingestion.ingest import ingest_and_chunk
from store.embedder import embed_text_units
from raptor.raptor_tree import build_raptor_tree
from indexing.indexer import index_into_qdrant
from store.vector_store import QdrantManager
from retrieval.pipeline import QueryPipeline

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


config = TREXConfig(
    qdrant_collection="trex_paper_collection",
)

# ============================================================
# INDEXING (run once per corpus)
# ============================================================
qdrant = QdrantManager(config)

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

# Step 3: Build RAPTOR tree
all_nodes = build_raptor_tree(text_units, config)

# Step 4: Index into Qdrant
index_into_qdrant(all_nodes, config)

# ============================================================
# QUERYING (run per user question)
# ============================================================
pipeline = QueryPipeline(config)

# OLTP-style (factual) — hits Level 0 chunks via BM25
result = pipeline.query("What was the performance upgrade of TREX over GraphRAG in terms of token count?")
print(result["answer"])

# OLAP-style (thematic) — hits Level 1-2 summaries via dense search
result = pipeline.query("What are the key strategic themes across all documents?")
print(result["answer"])

# Streaming
for token in pipeline.query_stream("How are the papers connected to each other?"):
    print(token, end="", flush=True)
