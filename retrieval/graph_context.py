"""
Query Step 3: Fetch neighborhood context from Neo4j.

For each matched entity, retrieve its 1-hop neighborhood:
- Connected entities and relationship descriptions
- ChunkRef IDs (for pulling graph-referenced chunks from Qdrant)

GraphRAG's mixed_context.py splits context budget into proportions:
  50% text units, 25% community reports, 25% entity/relationship tables

We do something simpler but equally effective:
- Graph context is formatted as readable text (not tables)
- Additional graph-referenced chunks are fetched from Qdrant if
  they weren't already in the top-K results
"""
import logging
from typing import Any

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TREXConfig
from store.graph_store import Neo4jManager

logger = logging.getLogger(__name__)


class GraphContextBuilder:
    """Fetch entity neighborhoods and format as readable context."""

    def __init__(self, config: TREXConfig, neo4j_manager: Neo4jManager):
        self.config = config
        self.neo4j = neo4j_manager

    def build(
        self,
        matched_entities: list[dict[str, Any]],
        max_relationships_per_entity: int = 10,
    ) -> dict[str, Any]:
        """
        For each matched entity, fetch its neighborhood from Neo4j.
        
        Returns:
            {
                "context_text": formatted string for the LLM prompt,
                "referenced_chunk_ids": set of chunk IDs reachable 
                    via MENTIONED_IN edges (for Step 4 enrichment)
            }
        """
        if not matched_entities:
            return {"context_text": "", "referenced_chunk_ids": set()}

        all_lines = ["### Knowledge Graph Context\n"]
        all_chunk_ids: set[str] = set()
        seen_entities = set()

        with self.neo4j.get_session() as session:
            for entity in matched_entities:
                name = entity["name"]
                if name in seen_entities:
                    continue
                seen_entities.add(name)

                # Entity header
                etype = entity.get("type", "Unknown")
                desc = entity.get("description", "")
                all_lines.append(f"**{name}** ({etype}): {desc}")

                # Fetch relationships
                result = session.run(
                    """
                    MATCH (e:Entity {name: $name})-[r:RELATES_TO]-(neighbor:Entity)
                    RETURN e.name AS entity, 
                           r.description AS rel_desc, 
                           r.weight AS weight,
                           neighbor.name AS neighbor, 
                           neighbor.type AS neighbor_type
                    ORDER BY r.weight DESC
                    LIMIT $limit
                    """,
                    name=name,
                    limit=max_relationships_per_entity,
                )

                for record in result:
                    all_lines.append(
                        f"  → {record['rel_desc']} → "
                        f"**{record['neighbor']}** ({record['neighbor_type']}) "
                        f"[weight: {record['weight']:.0f}]"
                    )

                # Fetch chunk provenance (MENTIONED_IN edges)
                chunk_result = session.run(
                    """
                    MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(c:ChunkRef)
                    RETURN c.chunk_id AS chunk_id
                    """,
                    name=name,
                )
                for record in chunk_result:
                    all_chunk_ids.add(record["chunk_id"])

                all_lines.append("")  # blank line between entities

        context_text = "\n".join(all_lines)
        logger.info(
            f"[GraphContext] Built context for {len(seen_entities)} entities, "
            f"found {len(all_chunk_ids)} referenced chunks"
        )

        return {
            "context_text": context_text,
            "referenced_chunk_ids": all_chunk_ids,
        }