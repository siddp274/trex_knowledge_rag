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
import json
from openai import OpenAI
from typing import Any, Optional

import tiktoken


@dataclass
class ChunkingConfig:
    """
    Mirrors GraphRAG's ChunkingConfig (Pydantic → dataclass for simplicity).
    """
    size: int = 1000

    overlap: int = 150

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
        set_transform: Optional[bool] = True,
    ) -> list[dict[str, Any]]:
        """
        Split text into token-based chunks.
        
        Args:
            text: The full document text to chunk.
            set_transform: Whether to apply the transformation
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
        transform = build_transformer(doc=text, enrich_fn=enrich_chunk, 
                                      count_tokens_fn=self.count_tokens)
        
        for i, chunk_text in enumerate(raw_chunks):
            transform_chunk = {
                "text": chunk_text,
                "original_text": chunk_text,
                "n_tokens": self.count_tokens(chunk_text),
                "index": i,
            }
            if set_transform:
                transform_chunk = transform(chunk_text)
                transform_chunk["index"] = i

            results.append(transform_chunk)

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

# Building custom, not GraphRAG
# Note, transform is applied after chunking, but context can be document scoped.
def enrich_chunk(doc:str, text: str) -> dict[str, Any]:

    system_prompt = """
You are given a document and a chunk from it. Your task is to generate a contextual representation of the chunk.
Return ONLY valid JSON (no markdown, no explanation).

Schema:
{
  "context": "string (50–100 tokens explaining what the chunk is about and its role in the document)",
  "document_section": "string (section or subsection name) if not present create your own based on the content",
  "chunk_role": "string (definition | evidence | conclusion | example | argument | other)",
  "entities": List["string"]
}

Rules:
- context must make the chunk self-contained for retrieval
- do not copy chunk verbatim
- extract only entities explicitly implied in chunk or present in the document.
"""

    user_prompt = f"""Document: {doc} Chunk:{text}"""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {
            "context": "json parsing failed, returning empty context",
            "document_section": "parsing failed",
            "chunk_role": "",
            "entities": []
        }


def build_transformer(
    doc:str,
    enrich_fn: Callable[[str, str], dict[str, Any]],
    count_tokens: Callable[[str], int],
):
    """
    enrich_fn: LLM function that returns structured context
    count_tokens: tokenizer function
    """

    def transform_chunk(text: str, index: int) -> dict[str, Any]:

        # 1. Get structured enrichment from LLM
        enrichment = enrich_fn(doc, text)

        context = enrichment.get("context", "")
        section = enrichment.get("document_section", "")
        role = enrichment.get("chunk_role", "")
        entities = enrichment.get("entities", [])

        # 2. Build embedding text (Anthropic-style)
        embedding_text = (
            f"Section: {section}\n"
            f"Role: {role}\n"
            f"Entities: {', '.join(entities)}\n\n"
            f"{context}\n\n"
            f"{text}"
        ).strip()

        # 3. Final unified chunk object
        return {
            "text": embedding_text,
            "original_text": text,
            "n_tokens": count_tokens(embedding_text),
            "index": index,

            # enriched metadata (flat, as you want)
            "context": context,
            "document_section": section,
            "chunk_role": role,
            "entities": entities,
        }

    return transform_chunk