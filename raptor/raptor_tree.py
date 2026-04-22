"""
Step 4: Build the RAPTOR Summary Tree (truncated at 2 levels per TREX).

This is the hierarchical clustering + summarization pipeline:
  Level 0: Leaf chunks (from Step 1)
  Level 1: Cluster summaries (5-15 related chunks → 1 summary)
  Level 2: Top-level summaries (3-10 Level 1 summaries → 1 overview)

At each level:
1. Take current nodes' embeddings
2. UMAP reduce dimensions (10D for L1, 2D for L2)
3. GMM + BIC find optimal cluster count
4. Soft-cluster nodes (a node can be in multiple clusters)
5. For each cluster, LLM-summarize the combined text
6. Embed the summary → becomes a new node

The TREX paper truncates at 2 levels because experiments showed
"2 levels gives 90% of quality at ~30% of full RAPTOR cost."

Why this matters for retrieval:
- OLTP queries ("What was Q4 revenue?") hit Level 0 chunks via BM25
- OLAP queries ("What are the strategic themes?") hit Level 1-2 via dense search
- The tree structure means a single top-level summary can pull in
  context from dozens of underlying chunks, enabling synthesis
  that flat vector search can't do.
"""
import logging
import time
from typing import Any

import numpy as np
from openai import OpenAI

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TREXConfig
from ingestion.text_unit import TextUnit
from ingestion.hashing import gen_sha512_hash
from store.embedder import TextUnitEmbedder
from raptor.clustering import reduce_dimensions, find_optimal_k, soft_cluster

logger = logging.getLogger(__name__)

SUMMARIZATION_PROMPT = """You are an expert research analyst.
Your task is to produce a dense, informative summary of the following text passages.

Rules:
- Preserve ALL key facts, entities, relationships, numbers, and dates
- Do not omit important details even if the text is long
- Write in clear, professional prose
- The summary should be self-contained — a reader should understand it 
  without seeing the original passages

Passages to summarize:

{text}"""


class RaptorTreeBuilder:
    """
    Builds a truncated RAPTOR tree over a list of TextUnits.
    
    Input: Level 0 TextUnits (with embeddings from Step 2)
    Output: Additional TextUnits at Level 1 and Level 2,
            with parent/children links to form the tree structure.
    """

    def __init__(self, config: TREXConfig):
        self.config = config
        self.llm_client = OpenAI(base_url=config.openai_indexer_endpoint, api_key=config.openai_indexer_api_key)
        self.embedder = TextUnitEmbedder(config)

    def build(self, text_units: list[TextUnit]) -> list[TextUnit]:
        """
        Entry point. Takes leaf TextUnits, returns ALL nodes 
        (leaves + summaries at each level).
        
        The returned list contains the original leaf nodes (unchanged)
        plus new summary nodes. The tree structure is encoded via
        parent_id and children_ids on each node.
        """
        all_nodes = list(text_units)
        current_level_nodes = [tu for tu in text_units if tu.level == 0]

        logger.info(
            f"[RAPTOR] Building tree over {len(current_level_nodes)} "
            f"leaf nodes (max {self.config.max_tree_levels} levels)"
        )

        for level in range(1, self.config.max_tree_levels + 1):
            if len(current_level_nodes) < self.config.min_cluster_size:
                logger.info(
                    f"[RAPTOR] Level {level}: Only {len(current_level_nodes)} nodes, "
                    f"need at least {self.config.min_cluster_size}. Stopping."
                )
                break

            summary_nodes = self._build_level(current_level_nodes, level)

            if not summary_nodes:
                logger.info(f"[RAPTOR] Level {level}: No summaries generated. Stopping.")
                break

            all_nodes.extend(summary_nodes)
            logger.info(
                f"[RAPTOR] Level {level}: Created {len(summary_nodes)} summary nodes "
                f"from {len(current_level_nodes)} input nodes"
            )
            current_level_nodes = summary_nodes

        total_by_level = {}
        for node in all_nodes:
            total_by_level[node.level] = total_by_level.get(node.level, 0) + 1
        logger.info(f"[RAPTOR] Tree complete. Nodes by level: {total_by_level}")

        return all_nodes

    def _build_level(
        self,
        nodes: list[TextUnit],
        level: int,
    ) -> list[TextUnit]:
        """
        Build one level of the tree:
        1. UMAP reduce the input node embeddings
        2. BIC → optimal k
        3. GMM soft cluster
        4. Summarize each cluster → new TextUnit
        5. Embed each summary
        """
        # --- Collect embeddings ---
        embeddings = np.array([node.embedding for node in nodes])

        # --- UMAP dimensionality reduction ---
        # Level 1: 10D (fine-grained topics within a document)
        # Level 2: 2D (broad themes across documents)
        n_components = (
            self.config.umap_components_l1 if level == 1
            else self.config.umap_components_l2
        )
        n_neighbors = min(15, len(nodes) - 1)
        if len(nodes) <= n_components + 1 or n_neighbors < 2:
            logger.info(f"[RAPTOR] Level {level}: Too few nodes ({len(nodes)}) for UMAP. Treating as single cluster.")
            # Skip clustering — summarize everything as one cluster
            reduced = None
        else:
            reduced = reduce_dimensions(embeddings, n_components=n_components, n_neighbors=n_neighbors)

        # --- Find optimal cluster count ---
        if reduced is None:
            cluster_assignments = {0: list(range(len(nodes)))}
        else:
            # --- Soft cluster ---
            k = find_optimal_k(reduced, max_k=min(self.config.max_clusters, len(nodes) - 1))
            cluster_assignments = soft_cluster(reduced, k=k, threshold=self.config.cluster_threshold)

        # --- Summarize each cluster ---
        summary_nodes = []

        for cluster_idx, member_indices in cluster_assignments.items():
            if len(member_indices) == 0:
                continue

            member_nodes = [nodes[i] for i in member_indices]

            # Build combined text for summarization
            # Use original_text (without prepended metadata) to avoid
            # repetitive "title: X" lines in the summary input
            combined_text = "\n\n---\n\n".join(
                node.original_text for node in member_nodes
            )

            # LLM summarization
            summary_text = self._summarize(combined_text)
            if not summary_text:
                continue

            # Embed the summary
            summary_embedding = self.embedder.embed_single(summary_text)

            # Build the summary TextUnit
            summary_id = gen_sha512_hash({"text": summary_text}, ["text"])

            # Source: inherit from the majority source in the cluster
            source_counts: dict[str, int] = {}
            for node in member_nodes:
                source_counts[node.source] = source_counts.get(node.source, 0) + 1
            primary_source = max(source_counts, key=source_counts.get)

            summary_node = TextUnit(
                id=summary_id,
                text=summary_text,
                original_text=summary_text,
                document_id=member_nodes[0].document_id,
                source=primary_source,
                title=f"Level-{level} Summary",
                n_tokens=self.embedder.config.chunk_size,  # approximate
                level=level,
                parent_id=None,             # filled if Level 3+ existed
                children_ids=[node.id for node in member_nodes],
                entity_ids=None,
                relationship_ids=None,
                embedding=summary_embedding,
            )

            # Recount tokens accurately
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            summary_node.n_tokens = len(enc.encode(summary_text))

            summary_nodes.append(summary_node)

            # Link children → parent
            for node in member_nodes:
                node.parent_id = summary_node.id

        return summary_nodes

    def _summarize(self, text: str) -> str:
        """
        Call the LLM to summarize clustered text.
        
        Uses the indexing model (gpt-4.1-mini) — cheap but good enough 
        for summarization. The RAPTOR paper used gpt-3.5-turbo for this.
        
        Truncates input if it exceeds model context. GPT-4.1-mini supports
        1M tokens so this is rarely hit, but defensive coding is good.
        """
        # Rough truncation safety — keep under 100K tokens
        # (text is already chunked, so clusters are typically 5K-30K tokens)
        max_chars = 400000  # ~100K tokens at ~4 chars/token
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated for length ...]"

        prompt = SUMMARIZATION_PROMPT.format(text=text)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.indexing_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"[RAPTOR] Summarization failed: {e}")
            return ""


def build_raptor_tree(
    text_units: list[TextUnit],
    config: TREXConfig,
) -> list[TextUnit]:
    """
    Step 4 entry point.
    
    Args:
        text_units: List from Steps 1-3 (Level 0 nodes with embeddings).
        config: Pipeline configuration.
    
    Returns:
        Extended list containing original Level 0 nodes PLUS
        new Level 1 and Level 2 summary nodes.
        All nodes have embeddings and tree linkage (parent_id, children_ids).
    """
    builder = RaptorTreeBuilder(config)
    return builder.build(text_units)