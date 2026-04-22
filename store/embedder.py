# ingestion/embedder.py
"""
Step 2: Embed all TextUnits.

Takes the TextUnit list from Step 1 and fills in the `embedding` field
on each unit using OpenAI's text-embedding-3-small.

Design decisions:
- Batch in groups of config.embedding_batch_size (default 100)
  to stay within API rate limits while minimizing round trips
- Uses the `text` field (which includes prepended metadata from Step 1),
  NOT `original_text` — so the embedding captures "this chunk is from 
  the Q4 Report" context
- Embeddings are also stored on the TextUnit objects in memory, 
  to be used in Step 5 (RAPTOR clustering via UMAP + GMM)
  and Step 6 (Qdrant indexing)
"""
import logging
import time
from openai import OpenAI, RateLimitError, APITimeoutError

from config import TREXConfig
from ingestion.text_unit import TextUnit

logger = logging.getLogger(__name__)


class TextUnitEmbedder:
    """
    Embeds TextUnits using OpenAI's embedding API.
    
    Why OpenAI's API directly instead of LangChain's OpenAIEmbeddings?
    - Direct control over batching and retry logic
    - LangChain's wrapper adds overhead but no value for this simple call
    - We need the raw float arrays for UMAP/GMM in Step 5 anyway
    """

    def __init__(self, config: TREXConfig):
        self.config = config
        self.client = OpenAI(
            base_url = config.openai_embedder_api_endpoint,
            api_key = config.openai_embedder_api_key,
        )
        self.model = config.embedding_model
        self.batch_size = config.embedding_batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts, handling batching and retries.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            List of embedding vectors (each is list[float] of length 1536).
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

            if (i + self.batch_size) % 500 == 0 or i + self.batch_size >= len(texts):
                logger.info(
                    f"[Embed] Progress: {min(i + self.batch_size, len(texts))}"
                    f"/{len(texts)} texts embedded"
                )

        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text. Used for query embedding and summary nodes."""
        return self._embed_batch_with_retry([text])[0]

    def _embed_batch_with_retry(
        self,
        texts: list[str],
        max_retries: int | None = None,
    ) -> list[list[float]]:
        """
        Call the OpenAI embedding API with exponential backoff.
        
        Handles:
        - RateLimitError: back off exponentially (1s, 2s, 4s...)
        - APITimeoutError: retry with same backoff
        - Other errors: raise immediately
        """
        retries = max_retries or self.config.max_retries
        backoff = 1.0

        for attempt in range(retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                # Response comes back sorted by index, but let's be safe
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]

            except (RateLimitError, APITimeoutError) as e:
                if attempt == retries:
                    raise
                logger.warning(
                    f"[Embed] {type(e).__name__} on attempt {attempt + 1}/{retries + 1}. "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2  # exponential backoff

    def estimate_cost(self, text_units: list[TextUnit]) -> float:
        """
        Estimate embedding cost in USD.
        text-embedding-3-small: $0.02 per 1M tokens.
        """
        total_tokens = sum(tu.n_tokens for tu in text_units)
        return total_tokens * 0.02 / 1_000_000


def embed_text_units(
    text_units: list[TextUnit],
    config: TREXConfig,
) -> list[TextUnit]:
    """
    Step 2 entry point: Embed all TextUnits in place.
    
    Fills the `embedding` field on each TextUnit.
    
    Args:
        text_units: List from Step 1 (embedding=None on each).
        config: Pipeline configuration.
    
    Returns:
        Same list with embedding field populated.
        (Mutates in place AND returns for chaining convenience.)
    """
    embedder = TextUnitEmbedder(config)

    # Cost estimate before we start
    est_cost = embedder.estimate_cost(text_units)
    logger.info(f"[Embed] Embedding {len(text_units)} text units")
    logger.info(f"[Embed] Estimated cost: ${est_cost:.4f}")

    # Extract texts — use the metadata-prepended `text` field
    texts = [tu.text for tu in text_units]

    # Batch embed
    embeddings = embedder.embed_texts(texts)

    # Attach embeddings back to TextUnits
    for tu, emb in zip(text_units, embeddings):
        tu.embedding = emb

    logger.info(f"[Embed] Done. All {len(text_units)} text units now have embeddings.")

    return text_units