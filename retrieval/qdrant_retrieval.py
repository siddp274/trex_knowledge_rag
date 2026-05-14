"""
Query Step 1: Hybrid retrieval from Qdrant (Dense + BM25 → RRF fusion).

Runs BOTH dense semantic search and BM25 keyword search simultaneously.
Qdrant fuses the two ranked lists using Reciprocal Rank Fusion (RRF)
internally when retrieval_mode=HYBRID.

RRF formula (Cormack et al. 2009):
    score(doc) = Σ 1/(k + rank_r(doc))  for each ranked list r
    k = 60 (smoothing constant)
"""
import logging
from typing import Any

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from openai import OpenAI

from config import TREXConfig
from store.vector_store import QdrantManager

logger = logging.getLogger(__name__)


class QdrantRetriever:
    """Hybrid search over the RAPTOR tree index in Qdrant."""

    def __init__(self, config: TREXConfig):
        self.config = config
        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )
        self.openai = OpenAI(base_url=config.openai_embedder_api_endpoint, api_key=config.openai_embedder_api_key)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Hybrid search: dense + BM25 fused via RRF.
        
        Returns top_k results, each a dict with:
        - text, original_text, level, source, title
        - text_unit_id, document_id, parent_id, children_ids
        - entity_ids, relationship_ids
        - score (RRF fusion score)
        """
        top_k = top_k or self.config.top_k

        dense_response = self.openai.embeddings.create(
            model=self.config.embedding_model,
            input=query,
        )
        dense_vector = dense_response.data[0].embedding

        sparse_list = list(self.sparse_model.embed([query]))
        sparse_vector = sparse_list[0]

        results = self.client.query_points(
            collection_name=self.config.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=QdrantManager.DENSE_VECTOR_NAME,
                    limit=top_k * 2,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices.tolist(),
                        values=sparse_vector.values.tolist(),
                    ),
                    using=QdrantManager.SPARSE_VECTOR_NAME,
                    limit=top_k * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )

        retrieved = []
        for point in results.points:
            result = {
                **point.payload,
                "score": point.score,
            }
            retrieved.append(result)

        level_counts = {}
        for r in retrieved:
            lvl = r.get("level", 0)
            level_counts[lvl] = level_counts.get(lvl, 0) + 1

        logger.info(
            f"[Qdrant] Retrieved {len(retrieved)} nodes "
            f"(by level: {level_counts})"
        )

        return retrieved
