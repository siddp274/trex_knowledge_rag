"""
Step 3c: Store entities and relationships in Neo4j + link back to TextUnits.

This writes the deduplicated, summarized entities and relationships into Neo4j
and creates the MENTIONED_IN provenance edges that bridge the knowledge graph
back to the vector-stored text chunks in Qdrant.

Also updates TextUnit objects with entity_ids and relationship_ids
(the forward link: chunk → entities it contains).
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

from store.graph_store import Neo4jManager
from ingestion.text_unit import TextUnit
from ingestion.hashing import gen_sha512_hash

logger = logging.getLogger(__name__)


def store_graph_in_neo4j(
    entities: list[dict],
    relationships: list[dict],
    neo4j_manager: Neo4jManager,
):
    """
    Write entities and relationships to Neo4j.
    
    Entity node properties:
    - name (UNIQUE constraint)
    - type
    - description (summarized)
    - aliases (list of alternative names from fuzzy dedup)
    - frequency (how many chunks mentioned this entity)
    
    Relationship edge properties:
    - description (summarized)
    - weight (sum of extraction weights = confidence)
    
    ChunkRef nodes + MENTIONED_IN edges:
    - Bridge from entities back to vector-stored chunks
    """
    logger.info(
        f"[Neo4j] Storing {len(entities)} entities and "
        f"{len(relationships)} relationships"
    )

    # --- Entities ---
    with neo4j_manager.get_session() as session:
        for entity in entities:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type          = $type,
                    e.description   = $description,
                    e.aliases       = $aliases,
                    e.frequency     = $frequency
                """,
                name=entity["title"],
                type=entity["type"],
                description=entity.get("description", ""),
                aliases=entity.get("aliases", []),
                frequency=entity.get("frequency", 1),
            )

            # Create MENTIONED_IN edges to ChunkRef nodes
            for chunk_id in entity.get("text_unit_ids", []):
                session.run(
                    """
                    MERGE (c:ChunkRef {chunk_id: $chunk_id})
                    WITH c
                    MATCH (e:Entity {name: $name})
                    MERGE (e)-[:MENTIONED_IN]->(c)
                    """,
                    chunk_id=chunk_id,
                    name=entity["title"],
                )

    logger.info(f"[Neo4j] Entities + provenance edges written")

    # --- Relationships ---
    with neo4j_manager.get_session() as session:
        for rel in relationships:
            session.run(
                """
                MATCH (a:Entity {name: $source})
                MATCH (b:Entity {name: $target})
                MERGE (a)-[r:RELATES_TO]->(b)
                SET r.description = $description,
                    r.weight      = $weight
                """,
                source=rel["source"],
                target=rel["target"],
                description=rel.get("description", ""),
                weight=rel.get("weight", 1.0),
            )

    logger.info(f"[Neo4j] Relationships written")


def link_entities_to_text_units(
    text_units: list[TextUnit],
    entities: list[dict],
    relationships: list[dict],
):
    """
    Update TextUnit objects with entity_ids and relationship_ids.
    
    This is the FORWARD link: TextUnit → entities it contains.
    (The REVERSE link is MENTIONED_IN in Neo4j: entity → chunks.)
    
    GraphRAG's TextUnit data model stores these as first-class fields,
    enabling queries like "which chunks mention entity X?" directly
    from the vector store metadata without hitting Neo4j.
    """
    # Build lookup: text_unit_id → list of entity titles
    tu_to_entities: dict[str, list[str]] = {}
    for entity in entities:
        for tu_id in entity.get("text_unit_ids", []):
            if tu_id not in tu_to_entities:
                tu_to_entities[tu_id] = []
            tu_to_entities[tu_id].append(entity["title"])

    # Build lookup: text_unit_id → list of relationship keys
    tu_to_rels: dict[str, list[str]] = {}
    for rel in relationships:
        rel_id = f"{rel['source']}→{rel['target']}"
        for tu_id in rel.get("text_unit_ids", []):
            if tu_id not in tu_to_rels:
                tu_to_rels[tu_id] = []
            tu_to_rels[tu_id].append(rel_id)

    # Update TextUnit objects
    linked_count = 0
    for tu in text_units:
        tu.entity_ids = tu_to_entities.get(tu.id, [])
        tu.relationship_ids = tu_to_rels.get(tu.id, [])
        if tu.entity_ids:
            linked_count += 1

    logger.info(
        f"[Link] {linked_count}/{len(text_units)} text units "
        f"linked to entities"
    )