"""Correctness tests."""

import tempfile
from pathlib import Path

import cphnsw
import numpy as np

from .conftest import SMALL_DIM, SMALL_K, SMALL_N, SMALL_NQUERY, brute_force_knn


def _compute_recall(result_ids, gt_ids, k):
    hits = len(set(result_ids[:k]) & set(gt_ids[:k]))
    return hits / k


class TestTranslationInvariance:
    def test_translation_invariance(self):
        rng = np.random.default_rng()
        base = rng.standard_normal((SMALL_N, SMALL_DIM), dtype=np.float32)
        queries = rng.standard_normal((SMALL_NQUERY, SMALL_DIM), dtype=np.float32)
        t = rng.standard_normal(SMALL_DIM, dtype=np.float32) * 10.0

        idx1 = cphnsw.Index(dim=SMALL_DIM)
        idx1.add(base)
        idx1.finalize()

        results1 = []
        for i in range(SMALL_NQUERY):
            ids, dists = idx1.search(queries[i], k=SMALL_K, recall_target=0.95)
            results1.append((ids.copy(), dists.copy()))

        base_shifted = base + t
        queries_shifted = queries + t

        idx2 = cphnsw.Index(dim=SMALL_DIM)
        idx2.add(base_shifted)
        idx2.finalize()

        results2 = []
        for i in range(SMALL_NQUERY):
            ids, dists = idx2.search(queries_shifted[i], k=SMALL_K, recall_target=0.95)
            results2.append((ids.copy(), dists.copy()))

        for i in range(SMALL_NQUERY):
            np.testing.assert_array_equal(
                results1[i][0], results2[i][0],
                err_msg=f"Query {i}: neighbor IDs differ after translation")
            np.testing.assert_allclose(
                results1[i][1], results2[i][1], rtol=1e-4, atol=1e-6,
                err_msg=f"Query {i}: distances differ after translation")


class TestSerializationRoundTrip:
    def test_save_load_round_trip(self):
        rng = np.random.default_rng()
        base = rng.standard_normal((SMALL_N, SMALL_DIM), dtype=np.float32)
        queries = rng.standard_normal((SMALL_NQUERY, SMALL_DIM), dtype=np.float32)

        idx1 = cphnsw.Index(dim=SMALL_DIM)
        idx1.add(base)
        idx1.finalize()

        results_before = []
        for i in range(SMALL_NQUERY):
            ids, dists = idx1.search(queries[i], k=SMALL_K, recall_target=0.95)
            results_before.append((ids.copy(), dists.copy()))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "test_index.bin")
            idx1.save(save_path)

            idx2 = cphnsw.load(save_path)

        results_after = []
        for i in range(SMALL_NQUERY):
            ids, dists = idx2.search(queries[i], k=SMALL_K, recall_target=0.95)
            results_after.append((ids.copy(), dists.copy()))

        for i in range(SMALL_NQUERY):
            np.testing.assert_array_equal(
                results_before[i][0], results_after[i][0],
                err_msg=f"Query {i}: IDs differ after save/load")
            np.testing.assert_allclose(
                results_before[i][1], results_after[i][1], rtol=1e-5, atol=1e-7,
                err_msg=f"Query {i}: distances differ after save/load")

    def test_basic_properties_preserved(self):
        rng = np.random.default_rng()
        base = rng.standard_normal((SMALL_N, SMALL_DIM), dtype=np.float32)

        idx1 = cphnsw.Index(dim=SMALL_DIM)
        idx1.add(base)
        idx1.finalize()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "test_index.bin")
            idx1.save(save_path)
            idx2 = cphnsw.load(save_path)

        assert idx2.size == idx1.size
        assert idx2.dim == idx1.dim
        assert idx2.is_finalized


class TestBasicFunctionality:
    def test_build_and_search(self, built_index, query_vectors):
        idx, base = built_index
        assert idx.size == SMALL_N
        assert idx.dim == SMALL_DIM
        assert idx.is_finalized

        ids, dists = idx.search(query_vectors[0], k=SMALL_K)
        assert len(ids) == SMALL_K
        assert len(dists) == SMALL_K
        assert all(d >= 0 for d in dists)

    def test_recall_above_threshold(self, built_index, query_vectors):
        idx, base = built_index
        gt_ids, _ = brute_force_knn(base, query_vectors, SMALL_K)

        recalls = []
        for i in range(SMALL_NQUERY):
            ids, _ = idx.search(query_vectors[i], k=SMALL_K, recall_target=0.95)
            recalls.append(_compute_recall(ids, gt_ids[i], SMALL_K))

        mean_recall = np.mean(recalls)
        assert mean_recall > 0.7, f"Mean recall too low: {mean_recall:.4f}"

    def test_batch_search(self, built_index, query_vectors):
        idx, _ = built_index
        ids, dists = idx.search_batch(query_vectors, k=SMALL_K)
        assert ids.shape == (SMALL_NQUERY, SMALL_K)
        assert dists.shape == (SMALL_NQUERY, SMALL_K)

    def test_calibration_info(self, built_index):
        idx, _ = built_index
        info = idx.calibration_info
        assert isinstance(info, dict)
        assert 'affine_a' in info
        assert 'calibrated' in info
        assert 'evt' in info
        assert info['calibrated'] is True
        assert info['evt']['fitted'] is True


class TestBFSReorderInvariance:
    def test_reorder_preserves_recall(self):
        rng = np.random.default_rng()
        base = rng.standard_normal((SMALL_N, SMALL_DIM), dtype=np.float32)
        queries = rng.standard_normal((SMALL_NQUERY, SMALL_DIM), dtype=np.float32)

        idx = cphnsw.Index(dim=SMALL_DIM)
        idx.add(base)
        idx.finalize()

        gt_ids, _ = brute_force_knn(base, queries, SMALL_K)

        recalls = []
        for i in range(SMALL_NQUERY):
            ids, _ = idx.search(queries[i], k=SMALL_K, recall_target=0.95)
            hits = len(set(ids[:SMALL_K]) & set(gt_ids[i][:SMALL_K]))
            recalls.append(hits / SMALL_K)

        mean_recall = np.mean(recalls)
        assert mean_recall > 0.6, f"Mean recall too low after reorder: {mean_recall:.4f}"
