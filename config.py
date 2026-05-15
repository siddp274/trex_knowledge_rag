import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TREXConfig:
    openai_embedder_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "sk-..."))
    openai_embedder_api_endpoint: str = field(default_factory=lambda: os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/"))

    openai_indexer_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "sk-..."))
    openai_indexer_endpoint: str = field(default_factory=lambda: os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/"))

    openai_query_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "sk-..."))
    openai_query_endpoint: str = field(default_factory=lambda: os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/"))

    context_feeding_model: str = "gpt-4.1-nano"
    indexing_model: str = "gpt-4.1-mini"
    query_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    qdrant_collection: str = "test_collection"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "my_api_key")
    qdrant_auto_create: bool = True

    chunk_size: int = 450 # tokens
    chunk_overlap: int = 90 # 20% overlap
    prepend_metadata: list[str] = field(default_factory=lambda: ["title"])

    max_tree_levels: int = 2  # As per the paper, level/depth 3 onwards we get marginal performance gains.
    cluster_threshold: float = 0.5
    umap_components_l1: int = 10
    umap_components_l2: int = 2
    max_clusters: int = 15
    min_cluster_size: int = 3

    top_k: int = 10
    rrf_k: int = 60  # Set by the RRF authors, can be tuned though.

    max_retries: int = 3
    embedding_batch_size: int = 100  # texts per API call to reduce calls and speed up embedding.


config = TREXConfig()
