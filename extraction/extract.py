# ingestion/extract.py
"""
Step 3 orchestrator: ties together extraction, merging, dedup, 
summarization, and Neo4j storage.

This is the single function you call from the main pipeline.
"""
import logging

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TREXConfig
from ingestion.text_unit import TextUnit
from store.graph_store import Neo4jManager
from extraction.graph_extractor import GraphExtractor
from extraction.graph_merger import (
    merge_extractions,
    fuzzy_deduplicate_entities,
    remap_relationships,
    summarize_descriptions,
)
from extraction.graph_builder import store_graph_in_neo4j, link_entities_to_text_units

logger = logging.getLogger(__name__)


def extract_and_store_graph(
    text_units: list[TextUnit],
    config: TREXConfig,
    neo4j_manager: Neo4jManager,
    entity_types: list[str] | None = None,
    max_gleanings: int = 1,
    similarity_threshold: float = 0.85,
) -> list[TextUnit]:
    """
    Full Step 3: Extract → Merge → Deduplicate → Summarize → Store.
    
    Args:
        text_units: List from Step 2 (with embeddings).
        config: Pipeline configuration.
        neo4j_manager: Initialized Neo4j connection.
        entity_types: Entity types to extract. Defaults to standard set.
        max_gleanings: Number of re-extraction passes per chunk. 
                       0 = no gleaning (fastest, cheapest).
                       1 = one pass (recommended — catches ~10% more entities).
        similarity_threshold: For fuzzy dedup. 0.85 = merge if cosine sim > 0.85.
    
    Returns:
        Same text_units list with entity_ids and relationship_ids populated.
    """
    # --- Gate: skip everything if graph is disabled ---
    if not config.enable_graph:
        logger.info(
            "[Step 3] Graph layer disabled (enable_graph=False). "
            "Skipping entity extraction. Indexing cost reduced ~60-70%."
        )
        return text_units

    if neo4j_manager is None:
        raise ValueError("neo4j_manager is required when enable_graph=True")
    
    # Only extract from Level 0 leaf chunks
    leaf_units = [tu for tu in text_units if tu.level == 0]
    logger.info(f"[Step 3] Extracting from {len(leaf_units)} leaf text units")

    # --- 3a: Extract raw entities + relationships from each chunk ---
    extractor = GraphExtractor(
        config=config,
        entity_types=entity_types,
        max_gleanings=max_gleanings,
    )

    all_extractions = []
    for i, tu in enumerate(leaf_units):
        if i % 20 == 0:
            logger.info(f"[Step 3a] Extracting {i}/{len(leaf_units)}...")
        result = extractor.extract_from_text_unit(tu)
        all_extractions.append(result)

    total_entities = sum(len(e["entities"]) for e in all_extractions)
    total_rels = sum(len(e["relationships"]) for e in all_extractions)
    logger.info(
        f"[Step 3a] Raw extraction: {total_entities} entity mentions, "
        f"{total_rels} relationship mentions"
    )

    # --- 3b-i: Exact-match merge (GraphRAG pattern) ---
    merged_entities, merged_relationships = merge_extractions(all_extractions)

    # --- 3b-ii: Fuzzy dedup via embeddings (our addition) ---
    deduped_entities, alias_map = fuzzy_deduplicate_entities(
        merged_entities, config, similarity_threshold
    )

    # Remap relationships to canonical entity names
    merged_relationships = remap_relationships(merged_relationships, alias_map)

    # --- 3b-iii: Summarize descriptions ---
    logger.info("[Step 3b] Summarizing descriptions...")
    deduped_entities = summarize_descriptions(deduped_entities, config)
    merged_relationships = summarize_descriptions(merged_relationships, config)

    # --- 3c: Store in Neo4j ---
    store_graph_in_neo4j(deduped_entities, merged_relationships, neo4j_manager)

    # --- 3d: Link entities back to TextUnits ---
    link_entities_to_text_units(text_units, deduped_entities, merged_relationships)

    logger.info(
        f"[Step 3] Complete: {len(deduped_entities)} entities, "
        f"{len(merged_relationships)} relationships stored in Neo4j"
    )

    return text_units