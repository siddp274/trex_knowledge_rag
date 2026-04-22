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
from store.vector_store import QdrantManager
from store.graph_store import Neo4jManager

# --- Configure for your project ---
config = TREXConfig(
    openai_embedder_api_endpoint=os.getenv("OPENAI_ENDPOINT"),
    openai_embedder_api_key=os.getenv("OPENAI_EMBEDDER_API_KEY"),
    qdrant_collection="sample_collection_trex",     
    neo4j_database="neo4j", 
    enable_graph=False    
)

# --- Auto-create infrastructure ---
qdrant = QdrantManager(config)  
if config.enable_graph: 
    neo4j = Neo4jManager(config)      

# --- Step 1: Ingest & Chunk ---
text_units = ingest_and_chunk(
    sources=["/Users/siddp278/Desktop/projects/graphRAG/papers/2404.16130v2.pdf"],
            # "/Users/siddp278/Desktop/projects/graphRAG/papers/2503.02922v1.pdf"],
    config=config,
)

# --- Step 2: Embed ---
text_units = embed_text_units(text_units, config)

# Every TextUnit now has:
#   .id            → deterministic SHA-512
#   .text          → "title: Annual Report 2024.\n<chunk>"
#   .embedding     → [0.0123, -0.0456, ...] (1536 floats)
#   .n_tokens      → exact count
#   .document_id   → links to source doc
#   .entity_ids    → None (filled in Step 3)
#   .level         → 0 (leaf)

print(f"""Sample TextUnit:\n
      ID: {text_units[0].id}\n
      Text: {text_units[0].text[:100]}...\n
      Embedding dim: {len(text_units[0].embedding)}\n
      Tokens: {text_units[0].n_tokens}""")

if config.enable_graph:
    neo4j.close()