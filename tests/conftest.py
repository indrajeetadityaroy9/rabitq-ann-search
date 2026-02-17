"""Shared fixtures."""

import cphnsw
import numpy as np
import pytest


SMALL_DIM = 32
SMALL_N = 500
SMALL_NQUERY = 20
SMALL_K = 10


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def random_vectors(rng):
    return rng.standard_normal((SMALL_N, SMALL_DIM), dtype=np.float32)


@pytest.fixture
def query_vectors(rng):
    return rng.standard_normal((SMALL_NQUERY, SMALL_DIM), dtype=np.float32)


@pytest.fixture
def built_index(random_vectors):
    idx = cphnsw.Index(dim=SMALL_DIM)
    idx.add(random_vectors)
    idx.finalize()
    return idx, random_vectors


def brute_force_knn(base, queries, k):
    n = base.shape[0]
    nq = queries.shape[0]
    ids = np.empty((nq, k), dtype=np.int64)
    dists = np.empty((nq, k), dtype=np.float32)
    for i in range(nq):
        d = np.sum((base - queries[i]) ** 2, axis=1)
        idx = np.argpartition(d, k)[:k]
        idx = idx[np.argsort(d[idx])]
        ids[i] = idx
        dists[i] = d[idx]
    return ids, dists
