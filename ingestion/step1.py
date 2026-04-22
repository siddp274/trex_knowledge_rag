import os
import sys

# Get absolute path of current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

# Add to sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from config import TREXConfig
from ingestion.ingest import ingest_and_chunk


config = TREXConfig(
        qdrant_collection="test_collection",
        neo4j_database="test_graph",
    )
text_units = ingest_and_chunk(
    sources=["/Users/siddp278/Desktop/projects/graphRAG/papers/2404.16130v2.pdf", 
                "/Users/siddp278/Desktop/projects/graphRAG/papers/2503.02922v1.pdf"],
    config=config,
)
for tu in text_units[:5]:
    print(f"ID: {tu.id}\nSource: {tu.source}\nTitle: {tu.title}\nTokens: {tu.n_tokens}\nText: {tu.text[:200]}...\n")