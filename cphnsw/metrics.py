"""ANN evaluation metrics."""

import numpy as np


def recall_at_k(results: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute mean recall@k for batched queries."""
    res = results[:, :k]
    gt = ground_truth[:, :k]
    hits = np.any(res[:, :, None] == gt[:, None, :], axis=2)
    return float(hits.sum(axis=1).mean()) / k
