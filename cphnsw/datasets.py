"""Dataset loaders."""

from pathlib import Path

import numpy as np


def load_fvecs(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    d = data[:1].view(np.int32)[0]
    return data.reshape(-1, d + 1)[:, 1:].copy()


def load_ivecs(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    k = int(data[0])
    return data.reshape(-1, k + 1)[:, 1:].copy()


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


def load_dataset(name: str, base_dir: str = "data/") -> dict:
    base_path = Path(base_dir) / name

    if name in FVECS_DATASETS:
        files = FVECS_DATASETS[name]
        base = load_fvecs(base_path / files["base"])
        queries = load_fvecs(base_path / files["queries"])
        groundtruth = load_ivecs(base_path / files["groundtruth"])
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
