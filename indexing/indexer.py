"""
Step 5: Index all RAPTOR tree nodes into Qdrant (hybrid: dense + BM25 sparse).

This is the final step of the indexing pipeline. It takes ALL nodes
(Level 0 leaves + Level 1-2 summaries) and stores them in Qdrant with:
- Dense vectors (text-embedding-3-small, already computed in Steps 2/4)
- Sparse vectors (BM25 via FastEmbed, computed here)
- Rich metadata (level, parent_id, children_ids, source, etc.)

At query time, Qdrant runs BOTH dense and sparse search simultaneously
and fuses results with Reciprocal Rank Fusion (RRF), giving us the
best of semantic similarity AND keyword matching in a single call.
"""
import logging
from typing import Any

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

from config import TREXConfig
from ingestion.text_unit import TextUnit
from store.vector_store import QdrantManager

logger = logging.getLogger(__name__)


class QdrantIndexer:
    """
    Indexes TextUnits into Qdrant with both dense and sparse vectors.
    
    Dense vectors: Already on each TextUnit from Step 2/4.
    Sparse vectors: Computed here via FastEmbed BM25.
    
    Why BM25 sparse vectors in addition to dense?
    - Dense search finds paraphrases and semantically similar text
    - BM25 finds exact keyword matches (names, acronyms, codes, numbers)
    - RRF fusion combines both without needing tuned weights
    
    Example where this matters:
    Query: "MSFT Q4 FY2024 revenue"
    - Dense search might match chunks about "Microsoft's financial performance"
    - BM25 matches the exact string "Q4 FY2024" and "revenue"
    - RRF ranks the chunk containing both the semantic match AND keywords highest
    """

    def __init__(self, config: TREXConfig):
        self.config = config
        self.qdrant_manager = QdrantManager(config)
        self.client = self.qdrant_manager.client
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def index_all(self, nodes: list[TextUnit], batch_size: int = 50):
        """
        Index all tree nodes into Qdrant.
        
        Each node becomes a Qdrant point with:
        - id: integer hash of the TextUnit ID (Qdrant needs int or UUID)
        - dense vector: from TextUnit.embedding
        - sparse vector: BM25 computed from TextUnit.text
        - payload: metadata for filtering and display at query time
        """
        collection = self.config.qdrant_collection
        total = len(nodes)

        for i in range(0, total, batch_size):
            batch = nodes[i : i + batch_size]
            self.client.upsert(
                collection_name=collection,
                points=self._build_points(batch),
            )
            indexed = min(i + batch_size, total)
            if indexed % 200 == 0 or indexed == total:
                logger.info(f"[Qdrant] Indexed {indexed}/{total} nodes")

        logger.info(f"[Qdrant] Indexing complete: {total} nodes")

    def _build_points(self, batch: list[TextUnit]) -> list[models.PointStruct]:
        """
        Convert a batch of TextUnits into Qdrant PointStructs.
        
        Each point has:
        - Named dense vector ("dense"): the OpenAI embedding
        - Named sparse vector ("sparse"): BM25 from FastEmbed
        - Payload: metadata dict for filtering and retrieval
        """
        texts = [node.text for node in batch]
        sparse_embeddings = list(self.sparse_model.embed(texts))

        points = []
        for node, sparse_emb in zip(batch, sparse_embeddings):
            point_id = int(node.id[:16], 16)
            payload = {
                "text_unit_id": node.id,
                "text": node.text,
                "original_text": node.original_text,
                "level": node.level,
                "source": node.source,
                "title": node.title,
                "document_id": node.document_id,
                "parent_id": node.parent_id,
                "children_ids": node.children_ids,
                "n_tokens": node.n_tokens,
                "context": node.context,
                "document_section": node.document_section,
                "chunk_role": node.chunk_role,
                "entities": node.entities,
            }
            point = models.PointStruct(
                id=point_id,
                vector={
                    QdrantManager.DENSE_VECTOR_NAME: node.embedding,
                    QdrantManager.SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                },
                payload=payload,
            )
            points.append(point)

        return points

    def get_collection_info(self) -> dict[str, Any]:
        """Return collection stats for verification."""
        info = self.client.get_collection(self.config.qdrant_collection)
        return {
            "name": self.config.qdrant_collection,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
        }


def index_into_qdrant(nodes: list[TextUnit], config: TREXConfig) -> dict[str, Any]:
    """
    Step 5 entry point.
    
    Args:
        nodes: All tree nodes from Step 4 (Level 0 + Level 1 + Level 2),
               each with embeddings already computed.
        config: Pipeline configuration.
    
    Returns:
        Collection info dict for verification.
    """
    indexer = QdrantIndexer(config)
    indexer.index_all(nodes)
    info = indexer.get_collection_info()
    logger.info(f"[Qdrant] Collection stats: {info}")
    return info
