"""ANN evaluation metrics."""

import numpy as np


def recall_at_k(results: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute mean recall@k for batched queries."""
    if len(results) == 0:
        return 0.0
    res = results[:, :k]
    gt = ground_truth[:, :k]
    hits = np.any(res[:, :, None] == gt[:, None, :], axis=2)
    return float(hits.sum(axis=1).mean()) / k


def average_distance_ratio(results: np.ndarray, ground_truth: np.ndarray,
                           base: np.ndarray, queries: np.ndarray, k: int) -> float:
    """Compute Average Distance Ratio (ADR) for batched queries.

    ADR = mean over queries of mean(dist(q, result_i) / dist(q, gt_i)).
    ADR >= 1.0; closer to 1.0 is better.
    """
    res_ids = results[:, :k].astype(np.int64)
    gt_ids = ground_truth[:, :k].astype(np.int64)
    res_dists = np.sum((base[res_ids] - queries[:, None, :]) ** 2, axis=2)
    gt_dists = np.sum((base[gt_ids] - queries[:, None, :]) ** 2, axis=2)
    return float(np.mean(res_dists / np.maximum(gt_dists, 1e-10)))
