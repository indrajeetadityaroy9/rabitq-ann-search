"""CP-HNSW adapters for VIBE."""

import cphnsw
import numpy as np

from ..base.module import BaseANN


class CPHNSW(BaseANN):
    """Flat index wrapper."""

    def __init__(self, metric, bits):
        self.bits = bits
        self.recall_target = 0.95

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        self.index = cphnsw.Index(dim=X.shape[1], bits=self.bits)
        self.index.add(X)
        self.index.finalize()

    def set_query_arguments(self, recall_target):
        self.recall_target = float(recall_target)

    def query(self, v, n):
        if v.dtype != np.float32:
            v = v.astype(np.float32)
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
        self.bits = bits
        self.recall_target = 0.95

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        self.index = cphnsw.HNSWIndex(dim=X.shape[1], bits=self.bits)
        self.index.add(X)
        self.index.finalize()

    def set_query_arguments(self, recall_target):
        self.recall_target = float(recall_target)

    def query(self, v, n):
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        ids, _ = self.index.search(v, k=n, recall_target=self.recall_target)
        return ids

    def __str__(self):
        return (
            f"CPHNSWHnsw(bits={self.bits}, "
            f"recall_target={self.recall_target:.2f})"
        )


class CPHNSWBatch(BaseANN):
    """Batch query wrapper."""

    def __init__(self, metric, bits):
        self.bits = bits
        self.recall_target = 0.95

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        self.index = cphnsw.Index(dim=X.shape[1], bits=self.bits)
        self.index.add(X)
        self.index.finalize()

    def set_query_arguments(self, recall_target):
        self.recall_target = float(recall_target)

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        ids, _ = self.index.search_batch(
            X, k=n, recall_target=self.recall_target
        )
        self.res = ids

    def get_batch_results(self):
        return [list(row[row >= 0]) for row in self.res]

    def __str__(self):
        return (
            f"CPHNSWBatch(bits={self.bits}, "
            f"recall_target={self.recall_target:.2f})"
        )
