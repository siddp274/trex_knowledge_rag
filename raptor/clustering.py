"""
Clustering utilities for RAPTOR tree construction.

Implements the UMAP dimensionality reduction + GMM soft clustering
pipeline from the RAPTOR paper, with BIC for automatic cluster count
selection.

Why this specific pipeline:
- Raw embeddings are 1536-dimensional. GMM struggles in high dimensions
  (curse of dimensionality — distances become meaningless).
- UMAP reduces to 10D (Level 1) or 2D (Level 2) while preserving 
  both local and global structure.
- GMM with soft assignment lets a chunk belong to multiple clusters.
  A chunk about "Google's AI healthcare initiatives" belongs in both
  the tech cluster and the healthcare cluster.
- BIC automatically selects the right number of clusters — no manual
  tuning needed.
"""
import logging
from typing import Any

import numpy as np
import umap.umap_ as umap
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce embedding dimensionality with UMAP.
    
    UMAP is imported lazily because it's slow to import (~2s)
    and not needed during query time.
    
    Args:
        embeddings: (n_samples, 1536) array of dense vectors.
        n_components: Target dimensions. 
                      10 for Level 1 (fine-grained distinction),
                      2 for Level 2 (broad themes).
        n_neighbors: Controls local vs global structure.
                     15 is the RAPTOR paper's default.
    
    Returns:
        (n_samples, n_components) reduced embeddings.
    """

    # Clamp to valid ranges
    n_samples = len(embeddings)
    n_components = min(n_components, n_samples - 2)
    n_neighbors = min(n_neighbors, n_samples - 2)

    if n_components < 1:
        return embeddings  # can't reduce further

    logger.info(
        f"[UMAP] Reducing {embeddings.shape} → "
        f"({n_samples}, {n_components}) with n_neighbors={n_neighbors}"
    )

    if n_samples <= 2:
        return embeddings

    if n_components < 1 or n_neighbors < 2:
        return embeddings
    try:
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, metric="cosine")
        return reducer.fit_transform(embeddings)
    
    except (TypeError, ValueError):
        return embeddings


def find_optimal_k(
    embeddings: np.ndarray,
    max_k: int = 15,
    min_k: int = 2,
) -> int:
    """
    Select optimal number of GMM clusters using BIC.
    
    BIC = ln(N) * k_params - 2 * ln(L_hat)
    Lower BIC = better model (penalizes complexity).
    
    From the RAPTOR paper: "We use BIC to select the model with the 
    optimal k, ensuring robust clustering performance."
    
    Args:
        embeddings: UMAP-reduced embeddings.
        max_k: Maximum clusters to try.
        min_k: Minimum clusters.
    
    Returns:
        Optimal k.
    """
    n_samples = len(embeddings)
    max_k = min(max_k, n_samples - 1)

    if max_k < min_k:
        return 1

    best_bic = np.inf
    best_k = min_k

    for k in range(min_k, max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42,
            )
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception:
            # GMM can fail if k is too large relative to data
            continue

    logger.info(f"[BIC] Optimal k={best_k} (BIC={best_bic:.2f})")
    return best_k


def soft_cluster(
    embeddings: np.ndarray,
    k: int,
    threshold: float = 0.5,
) -> dict[int, list[int]]:
    """
    GMM soft clustering — a node can belong to multiple clusters.
    
    From RAPTOR paper Section 3:
    "We use soft clustering, where nodes can belong to multiple clusters 
    without requiring a fixed number of clusters. This flexibility is 
    essential because individual text segments often contain information 
    relevant to various topics."
    
    Args:
        embeddings: UMAP-reduced embeddings.
        k: Number of clusters (from find_optimal_k).
        threshold: Minimum membership probability to assign a node 
                   to a cluster. 0.5 is the RAPTOR paper's default.
    
    Returns:
        Dict mapping cluster_idx → list of node indices.
        A node can appear in multiple clusters.
    """
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42,
    )
    gmm.fit(embeddings)

    # P(cluster_j | node_i): shape (n_samples, k)
    probs = gmm.predict_proba(embeddings)

    cluster_assignments: dict[int, list[int]] = {i: [] for i in range(k)}
    multi_assigned = 0

    for node_idx in range(len(embeddings)):
        assigned_clusters = []
        for cluster_idx in range(k):
            if probs[node_idx][cluster_idx] >= threshold:
                cluster_assignments[cluster_idx].append(node_idx)
                assigned_clusters.append(cluster_idx)
        if len(assigned_clusters) > 1:
            multi_assigned += 1

    # Remove empty clusters
    cluster_assignments = {
        k: v for k, v in cluster_assignments.items() if len(v) > 0
    }

    total_assignments = sum(len(v) for v in cluster_assignments.values())
    logger.info(
        f"[GMM] {len(cluster_assignments)} non-empty clusters, "
        f"{total_assignments} total assignments "
        f"({multi_assigned} nodes in multiple clusters)"
    )

    return cluster_assignments