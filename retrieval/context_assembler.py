"""
Query Step 4: Assemble final context from Qdrant results + graph context.

This merges the two retrieval paths:
1. Qdrant hybrid search results (ranked by RRF)
2. Graph-referenced chunks (discovered via entity → MENTIONED_IN → chunk)
3. Formatted graph context (entity descriptions + relationships)

All assembled under a token budget so we don't overflow the LLM context.

GraphRAG's approach (mixed_context.py) allocates fixed proportions:
  50% text, 25% community reports, 25% entity tables.

We use a priority-based approach instead:
  1. Qdrant RRF results first (already best-ranked)
  2. Graph-referenced chunks that weren't in top-K (fills gaps)
  3. Graph context text (structural knowledge)
This is simpler and more adaptive — if the query is factual (OLTP),
most budget goes to specific chunks; if thematic (OLAP), 
summaries and graph context naturally fill more space.
"""
import logging
from typing import Any
import tiktoken
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TREXConfig

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Merge vector + graph results under a token budget."""

    def __init__(self, config: TREXConfig):
        self.config = config
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def assemble(
        self,
        qdrant_results: list[dict[str, Any]],
        graph_context: dict[str, Any],
        max_context_tokens: int = 3500,
        max_graph_chunks: int = 3,
    ) -> dict[str, Any]:
        """
        Assemble final context for the answer generation prompt.
        
        Args:
            qdrant_results: From Step 1 (hybrid search).
            graph_context: From Step 3 (entity neighborhoods).
            max_context_tokens: Total token budget for passages + graph text.
            max_graph_chunks: Max additional chunks to pull via graph references.
        
        Returns:
            {
                "passages_text": formatted passages string,
                "graph_text": formatted graph context string,
                "total_tokens": actual tokens used,
                "num_passages": how many passages included,
                "levels_used": set of tree levels in the final context
            }
        """
        graph_text = graph_context.get("context_text", "")
        referenced_chunk_ids = graph_context.get("referenced_chunk_ids", set())

        # Reserve tokens for graph context
        graph_tokens = self._count_tokens(graph_text)
        passage_budget = max_context_tokens - graph_tokens

        if passage_budget < 200:
            # Graph context is too large, trim it
            graph_text = self._truncate_to_tokens(graph_text, max_context_tokens // 3)
            graph_tokens = self._count_tokens(graph_text)
            passage_budget = max_context_tokens - graph_tokens

        # --- Priority 1: Qdrant RRF results ---
        passages = []
        tokens_used = 0
        qdrant_chunk_ids = set()
        levels_used = set()

        for result in qdrant_results:
            passage = self._format_passage(result, len(passages) + 1)
            passage_tokens = self._count_tokens(passage)

            if tokens_used + passage_tokens > passage_budget:
                break

            passages.append(passage)
            tokens_used += passage_tokens
            qdrant_chunk_ids.add(result.get("text_unit_id", ""))
            levels_used.add(result.get("level", 0))

        # --- Priority 2: Graph-referenced chunks NOT already in results ---
        missing_ids = referenced_chunk_ids - qdrant_chunk_ids

        if missing_ids and tokens_used < passage_budget:
            # Find these chunks in our qdrant_results metadata
            # (they might have been in the oversampled prefetch but not top-K)
            # If not available, we note them but don't block
            graph_chunks_added = 0
            for result in qdrant_results:
                if result.get("text_unit_id") in missing_ids:
                    passage = self._format_passage(
                        result, len(passages) + 1, tag="Graph-Referenced"
                    )
                    passage_tokens = self._count_tokens(passage)

                    if tokens_used + passage_tokens > passage_budget:
                        break

                    passages.append(passage)
                    tokens_used += passage_tokens
                    levels_used.add(result.get("level", 0))
                    graph_chunks_added += 1

                    if graph_chunks_added >= max_graph_chunks:
                        break

            if graph_chunks_added > 0:
                logger.info(
                    f"[Context] Added {graph_chunks_added} graph-referenced chunks"
                )

        passages_text = "\n\n---\n\n".join(passages)
        total_tokens = tokens_used + graph_tokens

        logger.info(
            f"[Context] Assembled: {len(passages)} passages ({tokens_used} tokens) "
            f"+ graph context ({graph_tokens} tokens) = {total_tokens} total. "
            f"Levels used: {levels_used}"
        )

        return {
            "passages_text": passages_text,
            "graph_text": graph_text,
            "total_tokens": total_tokens,
            "num_passages": len(passages),
            "levels_used": levels_used,
        }

    def _format_passage(
        self,
        result: dict[str, Any],
        index: int,
        tag: str | None = None,
    ) -> str:
        """Format a single retrieved node as a labeled passage."""
        level = result.get("level", 0)
        level_label = "Raw Chunk" if level == 0 else f"Level-{level} Summary"
        source = result.get("source", "unknown")
        tag_str = f" | {tag}" if tag else ""

        text = result.get("original_text", result.get("text", ""))
        return f"[Passage {index} | {level_label} | Source: {source}{tag_str}]\n{text}"

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens]) + "\n[... truncated ...]"