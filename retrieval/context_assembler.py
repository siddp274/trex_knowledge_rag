"""
Query Step 3: Assemble final context from Qdrant results under a token budget.

Takes the RRF-ranked results from Qdrant and greedily fills a token budget,
formatting each result as a labeled passage for the answer generation prompt.

GraphRAG's approach (mixed_context.py) allocates fixed proportions across
text, community reports, and entity tables. We use a simpler priority-based
approach: Qdrant RRF results are already best-ranked, so we just pack as many
as fit within the budget, highest-ranked first.
"""
import logging
from typing import Any
import tiktoken

from config import TREXConfig

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Pack ranked Qdrant results into a token-budgeted context string."""

    def __init__(self, config: TREXConfig):
        self.config = config
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def assemble(
        self,
        qdrant_results: list[dict[str, Any]],
        max_context_tokens: int = 3500,
    ) -> dict[str, Any]:
        """
        Assemble final context for the answer generation prompt.
        
        Args:
            qdrant_results: From hybrid Qdrant search (RRF-ranked).
            max_context_tokens: Total token budget for all passages.
        
        Returns:
            {
                "passages_text": formatted passages string,
                "total_tokens": actual tokens used,
                "num_passages": how many passages included,
                "levels_used": set of tree levels in the final context
            }
        """
        passages = []
        tokens_used = 0
        levels_used = set()

        for result in qdrant_results:
            passage = self._format_passage(result, len(passages) + 1)
            passage_tokens = self._count_tokens(passage)

            if tokens_used + passage_tokens > max_context_tokens:
                break

            passages.append(passage)
            tokens_used += passage_tokens
            levels_used.add(result.get("level", 0))

        passages_text = "\n\n---\n\n".join(passages)

        logger.info(
            f"[Context] Assembled: {len(passages)} passages ({tokens_used} tokens). "
            f"Levels used: {levels_used}"
        )

        return {
            "passages_text": passages_text,
            "total_tokens": tokens_used,
            "num_passages": len(passages),
            "levels_used": levels_used,
        }

    def _format_passage(self, result: dict[str, Any], index: int) -> str:
        """Format a single retrieved node as a labeled passage."""
        level = result.get("level", 0)
        level_label = "Raw Chunk" if level == 0 else f"Level-{level} Summary"
        source = result.get("source", "unknown")
        text = result.get("original_text", result.get("text", ""))
        return f"[Passage {index} | {level_label} | Source: {source}]\n{text}"

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoder.encode(text))
