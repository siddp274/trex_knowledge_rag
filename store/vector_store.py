"""
Qdrant collection management with auto-creation.

Handles:
- Creating the collection if it doesn't exist (dense + sparse vectors)
- Checking if a collection exists
- Providing a configured QdrantClient for other modules to use

This module is used by both Step 2 (embedding/indexing) and 
Step 6 (full Qdrant indexing with BM25 sparse vectors).
"""
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

from config import TREXConfig

logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Manages Qdrant collection lifecycle.
    
    Auto-creates the collection with the correct vector config
    if it doesn't exist and auto_create is True.
    
    The collection has TWO vector types:
    - "dense": OpenAI text-embedding-3-small (1536 dims, cosine)
    - "sparse": BM25 via FastEmbed (for keyword matching)
    
    Both are needed for hybrid search with RRF fusion.
    """

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(self, config: TREXConfig):
        self.config = config
        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )
        if config.qdrant_auto_create:
            self._ensure_collection()

    def _ensure_collection(self) -> dict:
        """Ensure and then create collection if it doesn't exist. Skip if it does."""
        col = self.collection_exists()
        if col != None:
            logger.info(
                f"[Qdrant] Collection '{self.config.qdrant_collection}' "
                f"already exists — reusing"
            )
            return {
                    "collection": self.config.qdrant_collection, # has name in str
                    "points": col.points_count,
                    "status": col.status.value if col.status else "unknown",
                }

        logger.info(
            f"[Qdrant] Collection '{self.config.qdrant_collection}' "
            f"not found — creating"
        )
        self.client.create_collection(
            collection_name=self.config.qdrant_collection,
            vectors_config={
                self.DENSE_VECTOR_NAME: VectorParams(
                    size=self.config.embedding_dimensions,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info(
            f"[Qdrant] Created collection '{self.config.qdrant_collection}' "
            f"(dense: {self.config.embedding_dimensions}d cosine + sparse: BM25)"
        )

    def collection_exists(self) -> bool:
        """Check if the configured collection exists."""
        existing = [c.name for c in self.client.get_collections().collections]
        return self.client.get_collection(self.config.qdrant_collection) if self.config.qdrant_collection in existing else None
    
    def collection_list(self) -> list[str]:
        """List all collections in Qdrant."""
        collections = self.client.get_collections().collections
        collection_info = []
        for col in collections:
            info = self.client.get_collection(col.name)
            collection_info.append({
                "collection": col.name,
                "points": info.points_count,
                "status": info.status.value if info.status else "unknown",
            })
        return collection_info

    def delete_collection(self):
        """Delete the collection. Useful for full re-indexing."""
        self.client.delete_collection(self.config.qdrant_collection)
        logger.info(f"[Qdrant] Deleted collection '{self.config.qdrant_collection}'")
