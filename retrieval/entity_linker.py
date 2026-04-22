# retrieval/entity_linker.py
"""
Query Step 2: Extract entities from the user's question.

Uses the LLM to identify entity names in the query, then maps them
to entities in Neo4j.

GraphRAG's approach (mixed_context.py → map_query_to_entities):
- Embed the query
- Vector search against entity name/description embeddings
- Return top-k matched entities

Our approach combines:
- LLM extraction (catches implicit references like "Nadella" → SATYA NADELLA)
- Neo4j full-text search (fast, handles partial matches)
- Fallback to embedding similarity for misses
"""
import logging
from typing import Any

from openai import OpenAI
from neo4j import Session
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TREXConfig
from store.graph_store import Neo4jManager
from extraction.graph_extractor import DEFAULT_ENTITY_TYPES

logger = logging.getLogger(__name__)

class QueryEntityLinker:
    """Extract entities from query and link to Neo4j nodes."""

    def __init__(self, config: TREXConfig, neo4j_manager: Neo4jManager):
        self.config = config
        self.neo4j = neo4j_manager
        self.client = OpenAI(base_url=config.openai_indexer_endpoint, api_key=config.openai_indexer_api_key)

    def extract_and_link(self, query: str) -> list[dict[str, Any]]:
        """
        1. Ask LLM to extract entity names from the query
        2. Look each up in Neo4j (exact → fulltext → fuzzy)
        3. Return matched entity records with their descriptions
        """
        # Step 2a: LLM entity extraction from query
        entity_names = self._extract_entity_names(query)
        logger.info(f"[EntityLink] Extracted from query: {entity_names}")

        if not entity_names:
            return []

        # Step 2b: Link to Neo4j
        matched_entities = []
        for name in entity_names:
            entity = self._find_in_neo4j(name)
            if entity:
                matched_entities.append(entity)

        logger.info(
            f"[EntityLink] Linked {len(matched_entities)}/{len(entity_names)} "
            f"entities to Neo4j"
        )
        return matched_entities

    def _extract_entity_names(self, query: str) -> list[str]:
        """Use LLM to pull entity names from the query."""
        try:
            entity_types_str = ", ".join(DEFAULT_ENTITY_TYPES).lower()
            response = self.client.chat.completions.create(
                model=self.config.indexing_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Extract all entity names ({entity_types_str}) "
                        "from this question.\n"
                        "Return ONLY a comma-separated list of names, UPPERCASED.\n"
                        "If no entities found, return NONE.\n\n"
                        f"Question: {query}"
                    ),
                }],
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()

            if raw.upper() == "NONE" or not raw:
                return []

            names = [n.strip().upper() for n in raw.split(",") if n.strip()]
            return names

        except Exception as e:
            logger.warning(f"[EntityLink] LLM extraction failed: {e}")
            return []

    def _find_in_neo4j(self, name: str) -> dict[str, Any] | None:
        """
        Find entity in Neo4j. Try in order:
        1. Exact name match (or alias match)
        2. Full-text index search
        3. Case-insensitive substring match
        
        Returns the first match found, or None.
        """
        with self.neo4j.get_session() as session:
            # 1. Exact match on name or aliases
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name = $name OR $name IN e.aliases
                RETURN e.name AS name, e.type AS type, 
                       e.description AS description, e.frequency AS frequency
                LIMIT 1
                """,
                name=name,
            )
            record = result.single()
            if record:
                return dict(record)

            # 2. Full-text index (if available)
            try:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
                    YIELD node, score
                    WHERE score > 1.0
                    RETURN node.name AS name, node.type AS type,
                           node.description AS description, 
                           node.frequency AS frequency, score
                    ORDER BY score DESC
                    LIMIT 1
                    """,
                    query=name,
                )
                record = result.single()
                if record:
                    return dict(record)
            except Exception:
                pass  # fulltext index may not exist

            # 3. Substring fallback
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($name)
                RETURN e.name AS name, e.type AS type,
                       e.description AS description, e.frequency AS frequency
                ORDER BY e.frequency DESC
                LIMIT 1
                """,
                name=name,
            )
            record = result.single()
            if record:
                return dict(record)

        return None