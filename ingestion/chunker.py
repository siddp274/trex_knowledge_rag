"""
Token-aware text chunker using tiktoken, modeled after GraphRAG's TokenChunker.
GraphRAG defaults: chunk_size=1200, overlap=100 — which we adopt.

Key difference from LangChain's RecursiveCharacterTextSplitter:
- LangChain counts characters. "500 characters" could be 100-600 tokens.
- We count tokens. "1200 tokens" is always 1200 tokens.
- Since LLM context windows and embedding costs are token-denominated, this gives us exact budget control.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import tiktoken


@dataclass
class ChunkingConfig:
    """
    Mirrors GraphRAG's ChunkingConfig (Pydantic → dataclass for simplicity).
    """
    size: int = 1200

    overlap: int = 100

    encoding_model: str = "cl100k_base"
    """Tiktoken encoding. cl100k_base is used by text-embedding-3-small and GPT-4 family models."""

    prepend_metadata: list[str] | None = None
    """Metadata fields from the source document to prepend to each chunk.
    e.g., ["title", "creation_date"] → chunk text becomes:
      'title: Annual Report 2024.\ncreation_date: 2024-03-15.\n<actual chunk>'
    This grounds entity extraction — the LLM sees context about WHERE 
    this text came from."""


class TokenChunker:
    """
    Splits text into fixed-size token chunks with overlap.
    
    Follows GraphRAG's split_text_on_tokens algorithm exactly:
    1. Encode full text into token IDs
    2. Slice token array into windows of `size` tokens
    3. Each window starts `size - overlap` tokens after the previous
    4. Decode each window back to text
    
    This is a sliding-window approach on tokens. It's simpler than
    sentence-aware splitting but guarantees exact token counts,
    which matters for embedding API limits and context budgeting.
    """

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self._encoder = tiktoken.get_encoding(self.config.encoding_model)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self._encoder.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._encoder.decode(tokens)

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string."""
        return len(self.encode(text))

    def chunk(
        self,
        text: str,
        transform: Callable[[str], str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Split text into token-based chunks.
        
        Args:
            text: The full document text to chunk.
            transform: Optional function applied to each chunk's text
                       (e.g., prepend metadata). Applied AFTER splitting
                       so metadata doesn't eat into chunk boundaries,
                       but the transformed text is what gets embedded
                       and fed to entity extraction.
        
        Returns:
            List of dicts with keys: text, original_text, n_tokens, index
        """
        if not text or not text.strip():
            return []

        raw_chunks = self._split_on_tokens(text)
        results = []

        for i, chunk_text in enumerate(raw_chunks):
            original = chunk_text
            if transform:
                chunk_text = transform(chunk_text)

            results.append({
                "text": chunk_text,
                "original_text": original,
                "n_tokens": self.count_tokens(chunk_text),
                "index": i,
            })

        return results

    def _split_on_tokens(self, text: str) -> list[str]:
        """
        Core splitting algorithm — identical to GraphRAG's 
        split_single_text_on_tokens().
        
        Operates on token IDs directly:
        - Encode entire text into token array
        - Slide a window of `size` tokens
        - Step forward by `size - overlap` tokens each iteration
        - Decode each window back to text
        """
        input_ids = self.encode(text)
        chunk_size = self.config.size
        chunk_overlap = self.config.overlap

        result = []
        start_idx = 0
        cur_idx = min(start_idx + chunk_size, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

        while start_idx < len(input_ids):
            chunk_text = self.decode(list(chunk_ids))
            result.append(chunk_text)

            if cur_idx == len(input_ids):
                break

            start_idx += chunk_size - chunk_overlap
            cur_idx = min(start_idx + chunk_size, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]

        return result


def add_metadata_transformer(
    metadata: dict[str, Any],
    delimiter: str = ": ",
    line_delimiter: str = ".\n",
) -> Callable[[str], str]:
    """
    Creates a transform function that prepends metadata to chunk text.
    Adopted from GraphRAG's transformers.add_metadata().
    
    Example:
        transformer = add_metadata_transformer({"title": "Q4 Report", "date": "2024-12"})
        transformer("Revenue grew 15%...")
        → "title: Q4 Report.\ndate: 2024-12.\nRevenue grew 15%..."
    
    Why this matters:
        Without metadata, a chunk saying "Revenue grew 15%" is ambiguous —
        which company? which quarter? Prepending "title: Q4 Earnings Report"
        gives the entity extractor and embedder critical grounding context.
    """
    def transformer(text: str) -> str:
        metadata_str = line_delimiter.join(
            f"{k}{delimiter}{v}" for k, v in metadata.items()
        ) + line_delimiter
        return metadata_str + text

    return transformer