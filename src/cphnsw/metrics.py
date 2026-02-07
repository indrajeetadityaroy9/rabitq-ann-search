"""Evaluation metrics for approximate nearest neighbor search."""

import numpy as np


def recall_at_k(results: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k.

    Args:
        results: (k,) array of returned neighbor IDs.
        ground_truth: (k_gt,) array of true neighbor IDs.
        k: Number of neighbors to evaluate.

    Returns:
        Fraction of true top-k neighbors found in results.
    """
    gt_set = set(ground_truth[:k].tolist())
    hits = sum(1 for r in results[:k] if int(r) in gt_set)
    return hits / len(gt_set) if gt_set else 0.0


def recall_at_k_batch(results: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute mean recall@k over a batch of queries.

    Args:
        results: (n_queries, k) array of returned neighbor IDs.
        ground_truth: (n_queries, k_gt) array of true neighbor IDs.
        k: Number of neighbors to evaluate.

    Returns:
        Mean recall@k across all queries.
    """
    total = 0.0
    for i in range(len(results)):
        total += recall_at_k(results[i], ground_truth[i], k)
    return total / len(results) if len(results) > 0 else 0.0


def memory_usage(index) -> dict:
    """Estimate memory usage of an index.

    Args:
        index: A cphnsw.Index instance.

    Returns:
        Dict with keys: n_vectors, dim, estimated_bytes, per_vector_bytes.
    """
    n = index.size
    dim = index.dim
    # Each vector: D/8 bytes binary code + 32 neighbor IDs (4B each)
    #   + 32 packed code blocks + aux data + raw vector
    code_bytes = dim  # padded dim / 8 bits per byte (conservative)
    neighbor_bytes = 32 * 4  # 32 neighbor IDs
    aux_bytes = 32 * 12  # 32 * (3 floats)
    raw_vector_bytes = dim * 4  # stored for reranking
    per_vector = code_bytes + neighbor_bytes + aux_bytes + raw_vector_bytes
    total = n * per_vector

    return {
        "n_vectors": n,
        "dim": dim,
        "estimated_bytes": total,
        "per_vector_bytes": per_vector,
    }


def qps(latencies: np.ndarray) -> float:
    """Compute queries per second from latency array.

    Args:
        latencies: Array of per-query latencies in seconds.

    Returns:
        Queries per second.
    """
    mean_latency = np.mean(latencies)
    return 1.0 / mean_latency if mean_latency > 0 else 0.0
