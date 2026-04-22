import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TREXConfig:
    # --- OpenAI ---    
    openai_embedder_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDER_API_KEY", "sk-..."))
    openai_embedder_api_endpoint: str = field(default_factory=lambda: os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/"))
    
    openai_extractor_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_EXTRACTOR_API_KEY", "sk-..."))
    openai_extractor_endpoint: str = field(default_factory=lambda: os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/"))

    openai_indexer_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_INDEXER_API_KEY", "sk-..."))
    openai_indexer_endpoint: str = field(default_factory=lambda: os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/"))

    openai_query_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_QUERY_API_KEY", "sk-..."))
    openai_query_endpoint: str = field(default_factory=lambda: os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/"))

    indexing_model: str = "gpt-4.1-mini"
    query_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    extraction_model: str = "gpt-4.1-nano"
    embedding_dimensions: int = 1536

    # --- Qdrant ---
    qdrant_collection: str  = "test_collection"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "my_api_key")
    qdrant_auto_create: bool = True # auto-create if missing

    # --- Neo4j ---
    enable_graph: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "Shivjeet1"
    neo4j_database: str = "neo4j"              
    neo4j_auto_create: bool = True # auto-create if missing

    # --- Chunking ---
    chunk_size: int = 1200
    chunk_overlap: int = 100
    prepend_metadata: list[str] = field(default_factory=lambda: ["title"])

    # --- RAPTOR Tree ---
    max_tree_levels: int = 2 # As per the paper, level/depth 3 onwards we get marginal performance gains.
    cluster_threshold: float = 0.5
    umap_components_l1: int = 10
    umap_components_l2: int = 2
    max_clusters: int = 15
    min_cluster_size: int = 3

    # --- Retrieval ---
    top_k: int = 10
    rrf_k: int = 60 # Set by the RRF authors, can be tuned though.

    # --- LLM ---
    max_retries: int = 3
    embedding_batch_size: int = 100 # texts per API call to reduce calls and speed up embedding.


config = TREXConfig()