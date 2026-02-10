"""CP-HNSW: RaBitQ approximate nearest neighbor search for VIBE benchmark."""

import numpy as np

from ..base.module import BaseANN


class CPHNSW(BaseANN):
    """CP-HNSW with flat graph (Vamana-refined, RaBitQ 1-bit quantization)."""

    def __init__(self, metric, M, ef_construction, bits):
        if metric not in ("euclidean", "cosine", "normalized"):
            raise NotImplementedError(f"Metric {metric} not supported by CP-HNSW")
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.bits = bits

    def fit(self, X):
        import cphnsw

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            X = X / norms
        self.index = cphnsw.Index(
            dim=X.shape[1],
            M=self.M,
            ef_construction=self.ef_construction,
            bits=self.bits,
        )
        self.index.add(X)
        self.index.finalize()

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, n):
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            v = v / (np.linalg.norm(v) + 1e-10)
        ids, _ = self.index.search(v, k=n, ef=self.ef, error_epsilon=1.9)
        return ids

    def __str__(self):
        return "CPHNSW(M=%d, ef_construction=%d, bits=%d, ef=%d)" % (
            self.M, self.ef_construction, self.bits, self.ef,
        )


class CPHNSWHnsw(BaseANN):
    """CP-HNSW with multi-layer HNSW graph (RaBitQ quantization)."""

    def __init__(self, metric, M, ef_construction, bits):
        if metric not in ("euclidean", "cosine", "normalized"):
            raise NotImplementedError(f"Metric {metric} not supported by CP-HNSW")
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.bits = bits

    def fit(self, X):
        import cphnsw

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            X = X / norms
        self.index = cphnsw.HNSWIndex(
            dim=X.shape[1],
            M=self.M,
            ef_construction=self.ef_construction,
            bits=self.bits,
        )
        self.index.add(X)
        self.index.finalize()

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, n):
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            v = v / (np.linalg.norm(v) + 1e-10)
        ids, _ = self.index.search(v, k=n, ef=self.ef, error_epsilon=1.9)
        return ids

    def __str__(self):
        return "CPHNSWHnsw(M=%d, ef_construction=%d, bits=%d, ef=%d)" % (
            self.M, self.ef_construction, self.bits, self.ef,
        )


class CPHNSWGpu(BaseANN):
    """CP-HNSW GPU batch mode — parallel search_batch + GPU normalization."""

    def __init__(self, metric, M, ef_construction, bits):
        if metric not in ("euclidean", "cosine", "normalized"):
            raise NotImplementedError(f"Metric {metric} not supported by CP-HNSW")
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.bits = bits

    def fit(self, X):
        import cphnsw
        from cphnsw.gpu import gpu_normalize

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            X = gpu_normalize(X)
        self.index = cphnsw.Index(
            dim=X.shape[1],
            M=self.M,
            ef_construction=self.ef_construction,
            bits=self.bits,
        )
        self.index.add(X)
        self.index.finalize()
        self.num_points = X.shape[0]

    def set_query_arguments(self, ef):
        self.ef = ef

    def batch_query(self, X, n):
        from cphnsw.gpu import gpu_normalize

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric in ("cosine", "normalized"):
            X = gpu_normalize(X)
        ids, _ = self.index.search_batch(X, k=n, ef=self.ef, error_epsilon=1.9)
        self.res = ids

    def get_batch_results(self):
        return [list(row[row >= 0]) for row in self.res]

    def __str__(self):
        return "CPHNSWGpu(M=%d, ef_construction=%d, bits=%d, ef=%d)" % (
            self.M, self.ef_construction, self.bits, self.ef,
        )
