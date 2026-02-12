"""ANN evaluation metrics."""

import numpy as np


def recall_at_k(results: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k for one query."""
    gt_set = set(ground_truth[:k].tolist())
    hits = sum(1 for r in results[:k] if int(r) in gt_set)
    return hits / len(gt_set) if gt_set else 0.0


def recall_at_k_batch(results: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute mean recall@k for batched queries."""
    if len(results) == 0:
        return 0.0
    res = results[:, :k]
    gt = ground_truth[:, :k]
    hits = np.any(res[:, :, None] == gt[:, None, :], axis=2)
    return float(hits.sum(axis=1).mean()) / k


def memory_usage(index) -> dict:
    """Estimate index memory footprint."""
    n = index.size
    dim = index.dim
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
    """Compute queries-per-second from latency samples."""
    mean_latency = np.mean(latencies)
    return 1.0 / mean_latency if mean_latency > 0 else 0.0
