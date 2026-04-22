"""
Data model for text chunks (called TextUnits, following GraphRAG convention).
Each TextUnit is the atomic unit that flows through the entire pipeline:
  - gets embedded (Step 2)
  - gets entities extracted from it (Step 3)
  - gets indexed into Qdrant (Step 6)
  - gets linked back from Neo4j entities via MENTIONED_IN edges (Step 4)
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextUnit:
    """
    Mirrors GraphRAG's TextUnit data model with additions for our
    RAPTOR tree structure.
    
    GraphRAG stores entity_ids and relationship_ids directly on each chunk
    so you can always trace: chunk → entities it contains, and
    entity → chunks it appears in. We adopt this pattern.
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

    # --- Tree structure (filled in Step 5: RAPTOR tree building) ---
    level: int = 0
    """0 = leaf chunk, 1 = cluster summary, 2 = top-level summary."""

    parent_id: str | None = None
    """ID of the parent summary node in the RAPTOR tree."""

    children_ids: list[str] = field(default_factory=list)
    """IDs of child nodes this summary was built from."""

    # --- Graph linkage (filled in Step 3: entity extraction) ---
    entity_ids: list[str] | None = None
    """Entity IDs extracted from this chunk. Filled during entity extraction.
    This is the forward link: chunk → entities."""

    relationship_ids: list[str] | None = None
    """Relationship IDs extracted from this chunk. Filled during entity extraction."""

    # --- Embedding (filled in Step 2) ---
    embedding: list[float] | None = None
    """Dense vector from text-embedding-3-small. 1536 dimensions."""

    # --- Extra metadata ---
    attributes: dict[str, Any] | None = None
    """Arbitrary metadata from the source (CSV columns, PDF page numbers, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/indexing."""
        return {
            "id": self.id,
            "text": self.text,
            "original_text": self.original_text,
            "document_id": self.document_id,
            "source": self.source,
            "title": self.title,
            "n_tokens": self.n_tokens,
            "level": self.level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "entity_ids": self.entity_ids,
            "relationship_ids": self.relationship_ids,
            "attributes": self.attributes,
            # embedding intentionally excluded — stored in Qdrant, not serialized
        }