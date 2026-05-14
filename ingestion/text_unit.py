"""
Data model for text chunks (called TextUnits, following GraphRAG convention).
Each TextUnit is the atomic unit that flows through the entire pipeline:
  - gets embedded (Step 2)
  - gets indexed into Qdrant (Step 3)
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextUnit:
    """
    Core data model for a single text chunk.

    Tracks provenance (source, document_id), position in the RAPTOR tree
    (level, parent_id, children_ids), and the dense embedding used for
    both UMAP clustering and Qdrant indexing.
    """

    # Deterministic SHA-512 hash of the text content. Same text always produces same ID — prevents duplicates on re-ingestion.
    id: str

    text: str

    original_text: str

    # SHA-512 hash of the source document. Links chunk → source.
    document_id: str
    # The file path of the document
    source: str

    title: str

    n_tokens: int

    # --- Tree structure (filled in Step 4: RAPTOR tree building) ---
    level: int = 0
    """0 = leaf chunk, 1 = cluster summary, 2 = top-level summary."""

    parent_id: str | None = None
    """ID of the parent summary node in the RAPTOR tree."""

    children_ids: list[str] = field(default_factory=list)
    """IDs of child nodes this summary was built from."""

    # --- Embedding (filled in Step 2) ---
    embedding: list[float] | None = None
    """Dense vector from text-embedding-3-small. 1536 dimensions."""

    # --- Extra metadata ---
    attributes: dict[str, Any] | None = None
    """Arbitrary metadata from the source (CSV columns, PDF page numbers, etc.)."""

    context: str | None = None
    """Additional context about the chunk, e.g., surrounding text, section headers, etc."""
    
    document_section: str | None = None
    """The section of the document this chunk came from, if applicable (e.g., "Introduction")."""

    chunk_role: str | None = None
    """The role of the chunk within the document (e.g., "summary", "body")."""

    entities: list[str] | None = None
    """List of named entities found in the chunk."""
