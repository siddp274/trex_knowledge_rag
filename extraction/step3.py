import os
import sys
from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TREXConfig
from ingestion.ingest import ingest_and_chunk
from store.embedder import embed_text_units
from extraction.extract import extract_and_store_graph
from store.vector_store import QdrantManager
from store.graph_store import Neo4jManager

# Kinda forgot the names for each model type - fallback to defaults for now - Its openai ikkkk
config = TREXConfig(
    qdrant_collection="sample_collection_trex",
    neo4j_database="neo4j",
    enable_graph=False
)

qdrant = QdrantManager(config)

# Step 1: Ingest & Chunk
text_units = ingest_and_chunk(
    sources=["/Users/siddp278/Desktop/projects/graphRAG/papers/2404.16130v2.pdf"],
    config=config,
)

# Step 2: Embed
text_units = embed_text_units(text_units, config)

if config.enable_graph:
    neo4j = Neo4jManager(config)
    # Step 3: Extract entities/relationships → Neo4j
    text_units = extract_and_store_graph(
        text_units=text_units,
        config=config,
        neo4j_manager=neo4j,
        max_gleanings=1,            # 1 re-check pass per chunk
        similarity_threshold=0.85,  # merge entities with >85% name similarity
    )

    # At this point:
    # - Neo4j has: Entity nodes, RELATES_TO edges, ChunkRef nodes, MENTIONED_IN edges
    # - Each TextUnit has: entity_ids, relationship_ids populated
    # - Ready for Step 4 (RAPTOR tree) and Step 5 (Qdrant indexing)

    neo4j.close()