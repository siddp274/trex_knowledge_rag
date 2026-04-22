"""
Step 3b: Merge and deduplicate extracted entities and relationships.

Follows GraphRAG's _merge_entities / _merge_relationships pattern:
- Entities are grouped by (title, type) — descriptions aggregated into lists
- Relationships are grouped by (source, target) — weights summed
- Orphan relationships (referencing non-existent entities) are filtered out

THEN we add what GraphRAG doesn't do:
- Embedding-based fuzzy dedup across similar entity names
- LLM-based description summarization for entities with many descriptions
"""
import logging
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from openai import OpenAI

import os
import sys
from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TREXConfig
from store.embedder import TextUnitEmbedder
from prompts.prompt import DESCRIPTION_SUMMARIZE_PROMPT

logger = logging.getLogger(__name__)


def merge_extractions(
    all_extractions: list[dict[str, Any]],
) -> tuple[list[dict], list[dict]]:
    """
    Step 3b-i: Exact-match merge (same as GraphRAG's _merge_entities).
    
    Groups entities by (title, type) and relationships by (source, target).
    Aggregates descriptions into lists, counts frequency.
    
    Args:
        all_extractions: List of extraction results from Step 3a.
            Each is {"source_id": str, "entities": [...], "relationships": [...]}
    
    Returns:
        (merged_entities, merged_relationships)
    """
    # --- Merge entities ---
    entity_map: dict[tuple[str, str], dict] = {}

    for extraction in all_extractions:
        source_id = extraction["source_id"]
        for entity in extraction["entities"]:
            key = (entity["title"], entity["type"])
            if key not in entity_map:
                entity_map[key] = {
                    "title": entity["title"],
                    "type": entity["type"],
                    "descriptions": [],
                    "text_unit_ids": [],
                    "frequency": 0,
                }
            entity_map[key]["descriptions"].append(entity["description"])
            entity_map[key]["text_unit_ids"].append(source_id)
            entity_map[key]["frequency"] += 1

    merged_entities = list(entity_map.values())

    # --- Merge relationships ---
    rel_map: dict[tuple[str, str], dict] = {}

    for extraction in all_extractions:
        source_id = extraction["source_id"]
        for rel in extraction["relationships"]:
            key = (rel["source"], rel["target"])
            if key not in rel_map:
                rel_map[key] = {
                    "source": rel["source"],
                    "target": rel["target"],
                    "descriptions": [],
                    "text_unit_ids": [],
                    "weight": 0.0,
                }
            rel_map[key]["descriptions"].append(rel["description"])
            rel_map[key]["text_unit_ids"].append(source_id)
            rel_map[key]["weight"] += rel["weight"]

    merged_relationships = list(rel_map.values())

    # --- Filter orphan relationships (GraphRAG pattern) ---
    entity_titles = {e["title"] for e in merged_entities}
    before_count = len(merged_relationships)

    merged_relationships = [
        r for r in merged_relationships
        if r["source"] in entity_titles and r["target"] in entity_titles
    ]

    dropped = before_count - len(merged_relationships)
    if dropped > 0:
        logger.warning(
            f"[Merge] Dropped {dropped} orphan relationship(s) "
            f"referencing non-existent entities"
        )

    logger.info(
        f"[Merge] Exact-match merge: {len(merged_entities)} entities, "
        f"{len(merged_relationships)} relationships"
    )

    return merged_entities, merged_relationships


def fuzzy_deduplicate_entities(
    entities: list[dict],
    config: TREXConfig,
    similarity_threshold: float = 0.85,
) -> tuple[list[dict], dict[str, str]]:
    """
    Step 3b-ii: Embedding-based fuzzy deduplication.
    
    GraphRAG only does exact uppercase matching. This catches cases like:
    - "GOOGLE" and "GOOGLE LLC" and "ALPHABET/GOOGLE"
    - "SATYA NADELLA" and "NADELLA"
    - "ARTIFICIAL INTELLIGENCE" and "AI" (if both extracted as type CONCEPT)
    
    Algorithm:
    1. Group entities by type (only merge within same type)
    2. Embed all entity names
    3. Agglomerative clustering with cosine distance
    4. For each cluster, pick the most-mentioned name as canonical
    5. Merge descriptions and text_unit_ids from all cluster members
    
    Returns:
        (deduplicated_entities, alias_map)
        alias_map: maps every original title → its canonical title
    """
    if len(entities) <= 1:
        alias_map = {e["title"]: e["title"] for e in entities}
        return entities, alias_map

    embedder = TextUnitEmbedder(config)
    alias_map: dict[str, str] = {}

    # Group by entity type
    type_groups: dict[str, list[dict]] = {}
    for entity in entities:
        etype = entity["type"]
        if etype not in type_groups:
            type_groups[etype] = []
        type_groups[etype].append(entity)

    deduped_entities = []

    for etype, group in type_groups.items():
        if len(group) <= 1:
            for e in group:
                alias_map[e["title"]] = e["title"]
            deduped_entities.extend(group)
            continue

        # Embed entity names (cheap — short strings)
        names = [e["title"] for e in group]
        embeddings = embedder.embed_texts(names)
        embedding_matrix = np.array(embeddings)

        # Agglomerative clustering with cosine distance
        # distance_threshold = 1 - similarity_threshold
        # cosine_distance = 1 - cosine_similarity
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embedding_matrix)

        # Group entities by cluster
        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        for cluster_indices in clusters.values():
            members = [group[i] for i in cluster_indices]

            # Canonical = the name with highest mention frequency
            canonical = max(members, key=lambda e: e["frequency"])
            canonical_title = canonical["title"]

            # Merge all members into canonical
            merged_descriptions = []
            merged_text_unit_ids = []
            merged_frequency = 0
            all_titles = []

            for member in members:
                merged_descriptions.extend(member["descriptions"])
                merged_text_unit_ids.extend(member["text_unit_ids"])
                merged_frequency += member["frequency"]
                all_titles.append(member["title"])
                alias_map[member["title"]] = canonical_title

            deduped_entity = {
                "title": canonical_title,
                "type": etype,
                "descriptions": merged_descriptions,
                "text_unit_ids": list(set(merged_text_unit_ids)),
                "frequency": merged_frequency,
                "aliases": all_titles if len(all_titles) > 1 else [],
            }
            deduped_entities.append(deduped_entity)

            if len(all_titles) > 1:
                logger.info(
                    f"[Dedup] Merged {all_titles} → '{canonical_title}'"
                )

    logger.info(
        f"[Dedup] Fuzzy dedup: {len(entities)} → {len(deduped_entities)} entities"
    )

    return deduped_entities, alias_map


def remap_relationships(
    relationships: list[dict],
    alias_map: dict[str, str],
) -> list[dict]:
    """
    After fuzzy dedup, remap relationship source/target to canonical names
    and re-merge any that now have the same (source, target) key.
    """
    # Remap
    for rel in relationships:
        rel["source"] = alias_map.get(rel["source"], rel["source"])
        rel["target"] = alias_map.get(rel["target"], rel["target"])

    # Remove self-loops created by merging
    relationships = [r for r in relationships if r["source"] != r["target"]]

    # Re-merge relationships that now share the same (source, target)
    rel_map: dict[tuple[str, str], dict] = {}
    for rel in relationships:
        key = (rel["source"], rel["target"])
        if key not in rel_map:
            rel_map[key] = {
                "source": rel["source"],
                "target": rel["target"],
                "descriptions": [],
                "text_unit_ids": [],
                "weight": 0.0,
            }
        rel_map[key]["descriptions"].extend(rel["descriptions"])
        rel_map[key]["text_unit_ids"].extend(rel["text_unit_ids"])
        rel_map[key]["weight"] += rel["weight"]

    remapped = list(rel_map.values())
    # Deduplicate text_unit_ids
    for rel in remapped:
        rel["text_unit_ids"] = list(set(rel["text_unit_ids"]))

    logger.info(
        f"[Dedup] After remap: {len(remapped)} relationships"
    )
    return remapped


def summarize_descriptions(
    items: list[dict],
    config: TREXConfig,
    max_descriptions_before_summarize: int = 3,
) -> list[dict]:
    """
    Step 3b-iii: Condense multiple descriptions into one per entity/relationship.
    
    GraphRAG does this as a separate workflow step (summarize_descriptions).
    We inline it here. Logic:
    - If <= max_descriptions_before_summarize unique descriptions, just join them.
    - If more, call the LLM to produce a single summary.
    
    This saves LLM calls for entities mentioned in only 1-2 chunks (majority),
    while properly summarizing heavily-mentioned entities.
    """
    client = OpenAI(
            base_url = config.openai_indexer_endpoint,
            api_key = config.openai_indexer_api_key,
        )

    for item in items:
        descriptions = item.get("descriptions", [])
        # Deduplicate identical descriptions
        unique_descs = list(dict.fromkeys(descriptions))

        if len(unique_descs) <= max_descriptions_before_summarize:
            item["description"] = " | ".join(unique_descs)
        else:
            # LLM summarization
            desc_list_text = "\n".join(
                f"- {desc}" for desc in unique_descs
            )
            prompt = DESCRIPTION_SUMMARIZE_PROMPT.format(
                description_list=desc_list_text
            )
            try:
                response = client.chat.completions.create(
                    model=config.indexing_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                item["description"] = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"[Summarize] Failed for {item.get('title', 'unknown')}: {e}")
                item["description"] = " | ".join(unique_descs[:3]) + "..."

    return items