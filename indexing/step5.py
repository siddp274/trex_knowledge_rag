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

config = TREXConfig(
    qdrant_collection="sample_collection_trex",
    neo4j_database="neo4j",
    enable_graph=False,
)

# Auto-create infrastructure
qdrant = QdrantManager(config)
neo4j = Neo4jManager(config)

# Step 1: Ingest & Chunk
text_units = ingest_and_chunk(
    sources=["/Users/siddp278/Desktop/projects/graphRAG/papers/2404.16130v2.pdf"],
    config=config,
)

# Step 2: Embed
text_units = embed_text_units(text_units, config)

# Step 3: Extract entities/relationships → Neo4j
if config.enable_graph:
    text_units = extract_and_store_graph(
        text_units=text_units,
        config=config,
        neo4j_manager=neo4j,
        max_gleanings=1,
)

# Step 4: Build RAPTOR tree
all_nodes = build_raptor_tree(text_units, config)

# Step 5: Index everything into Qdrant
collection_info = index_into_qdrant(all_nodes, config)

print(f"Indexing complete: {collection_info}")
if config.enable_graph:
    neo4j.close()