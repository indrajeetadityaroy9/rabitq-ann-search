"""CP-HNSW adapters for VIBE."""

import numpy as np

from ..base.module import BaseANN


def _validate_metric(metric: str) -> None:
    if metric not in ("euclidean", "cosine", "normalized"):
        raise NotImplementedError(
            f"Metric {metric} not supported by Configuration-Parameterless HNSW (CP-HNSW)"
        )


def _normalize_matrix_cpu(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return X / norms


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)

class CPHNSW(BaseANN):
    """Flat index wrapper."""

    def __init__(self, metric, bits):
        _validate_metric(metric)
        self.metric = metric
        self.bits = bits
        self.recall_target = 0.95

    def fit(self, X):
        import cphnsw

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            X = _normalize_matrix_cpu(X)
        self.index = cphnsw.Index(dim=X.shape[1], bits=self.bits)
        self.index.add(X)
        self.index.finalize()

    def set_query_arguments(self, recall_target):
        self.recall_target = float(recall_target)

    def query(self, v, n):
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            v = _normalize_vector(v)
        ids, _ = self.index.search(v, k=n, recall_target=self.recall_target)
        return ids

    def __str__(self):
        return (
            f"CPHNSW(bits={self.bits}, "
            f"recall_target={self.recall_target:.2f})"
        )


class CPHNSWHnsw(BaseANN):
    """Hierarchical index wrapper."""

    def __init__(self, metric, bits):
        _validate_metric(metric)
        self.metric = metric
        self.bits = bits
        self.recall_target = 0.95

    def fit(self, X):
        import cphnsw

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            X = _normalize_matrix_cpu(X)
        self.index = cphnsw.HNSWIndex(dim=X.shape[1], bits=self.bits)
        self.index.add(X)
        self.index.finalize()

    def set_query_arguments(self, recall_target):
        self.recall_target = float(recall_target)

    def query(self, v, n):
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            v = _normalize_vector(v)
        ids, _ = self.index.search(v, k=n, recall_target=self.recall_target)
        return ids

    def __str__(self):
        return (
            f"CPHNSWHnsw(bits={self.bits}, "
            f"recall_target={self.recall_target:.2f})"
        )


class CPHNSWGpu(BaseANN):
    """GPU batch wrapper."""

    def __init__(self, metric, bits):
        _validate_metric(metric)
        self.metric = metric
        self.bits = bits
        self.recall_target = 0.95

    def fit(self, X):
        import cphnsw
        from cphnsw.gpu import gpu_normalize

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            X = gpu_normalize(X)
        self.index = cphnsw.Index(dim=X.shape[1], bits=self.bits)
        self.index.add(X)
        self.index.finalize()

    def set_query_arguments(self, recall_target):
        self.recall_target = float(recall_target)

    def batch_query(self, X, n):
        from cphnsw.gpu import gpu_normalize

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            X = gpu_normalize(X)
        ids, _ = self.index.search_batch(
            X, k=n, recall_target=self.recall_target
        )
        self.res = ids

    def get_batch_results(self):
        return [list(row[row >= 0]) for row in self.res]

    def __str__(self):
        return (
            f"CPHNSWGpu(bits={self.bits}, "
            f"recall_target={self.recall_target:.2f})"
        )
