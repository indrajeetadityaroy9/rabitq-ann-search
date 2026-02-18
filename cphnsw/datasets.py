"""Dataset loaders."""

from pathlib import Path

import numpy as np

from cphnsw.paths import resolve_project_path


FVECS_DATASETS = {
    "sift1m": {
        "base": "sift_base.fvecs",
        "queries": "sift_query.fvecs",
        "groundtruth": "sift_groundtruth.ivecs",
    },
    "gist1m": {
        "base": "gist_base.fvecs",
        "queries": "gist_query.fvecs",
        "groundtruth": "gist_groundtruth.ivecs",
    },
}

NPY_DATASETS = {"openai1536", "msmarco10m"}

ALL_DATASETS = list(FVECS_DATASETS.keys()) + sorted(NPY_DATASETS)


def load_dataset(name: str, base_dir: Path | str) -> dict:
    base_path = resolve_project_path(base_dir) / name

    if name in FVECS_DATASETS:
        files = FVECS_DATASETS[name]
        base_raw = np.fromfile(base_path / files["base"], dtype=np.float32)
        base_dim = base_raw[:1].view(np.int32)[0]
        base = base_raw.reshape(-1, base_dim + 1)[:, 1:].copy()

        query_raw = np.fromfile(base_path / files["queries"], dtype=np.float32)
        query_dim = query_raw[:1].view(np.int32)[0]
        queries = query_raw.reshape(-1, query_dim + 1)[:, 1:].copy()

        gt_raw = np.fromfile(base_path / files["groundtruth"], dtype=np.int32)
        gt_k = int(gt_raw[0])
        groundtruth = gt_raw.reshape(-1, gt_k + 1)[:, 1:].copy()
    elif name in NPY_DATASETS:
        base = np.load(base_path / "base.npy").astype(np.float32)
        queries = np.load(base_path / "queries.npy").astype(np.float32)
        groundtruth = np.load(base_path / "groundtruth.npy").astype(np.int32)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Supported: {ALL_DATASETS}.")

    return {
        "base": base,
        "queries": queries,
        "groundtruth": groundtruth,
        "dim": base.shape[1],
    }
