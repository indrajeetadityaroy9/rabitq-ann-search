"""Dataset loaders and download helpers."""

import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


def _emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def load_fvecs(path: str) -> np.ndarray:
    """Load a `.fvecs` matrix."""
    data = np.fromfile(path, dtype=np.float32)
    d = data[:1].view(np.int32)[0]
    return data.reshape(-1, d + 1)[:, 1:].copy()


def load_ivecs(path: str) -> np.ndarray:
    """Load an `.ivecs` matrix."""
    data = np.fromfile(path, dtype=np.int32)
    k = int(data[0])
    return data.reshape(-1, k + 1)[:, 1:].copy()


def load_hdf5_dataset(path: str) -> dict:
    """Load a dataset from ANN-benchmarks HDF5 format."""
    import h5py
    with h5py.File(path, 'r') as f:
        base = np.array(f['train'])
        queries = np.array(f['test'])
        groundtruth = np.array(f['neighbors']).astype(np.int32)
    return {
        "base": base,
        "queries": queries,
        "groundtruth": groundtruth,
        "dim": base.shape[1],
    }


SIFT1M_URL = "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift.tar.gz"
GLOVE200_URL = "https://huggingface.co/datasets/qbo-odp/glove-200-angular/resolve/main/glove-200-angular.hdf5"


def download_sift1m(dest: str = "data/sift1m/"):
    """Download SIFT-1M if missing."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "sift_base.fvecs").exists():
        _emit("dataset_cache_hit", dataset="sift1m", path=str(dest))
        return

    tarball = dest / "sift.tar.gz"
    _emit("dataset_download_start", dataset="sift1m", url=SIFT1M_URL, path=str(tarball))
    urllib.request.urlretrieve(SIFT1M_URL, tarball)

    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                # Keep extracted filenames flat to match the loader layout.
                member.name = Path(member.name).name
                tar.extract(member, dest)

    tarball.unlink()
    _emit("dataset_ready", dataset="sift1m", path=str(dest))


def download_glove200(dest: str = "data/glove200/"):
    """Download GloVe-200 if missing."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    hdf5_path = dest / "glove-200-angular.hdf5"
    if hdf5_path.exists():
        _emit("dataset_cache_hit", dataset="glove200", path=str(hdf5_path))
        return

    _emit("dataset_download_start", dataset="glove200", url=GLOVE200_URL, path=str(hdf5_path))
    urllib.request.urlretrieve(GLOVE200_URL, hdf5_path)
    _emit("dataset_ready", dataset="glove200", path=str(hdf5_path))


def load_dataset(name: str, base_dir: str = "data/") -> dict:
    """Load a named benchmark dataset or HDF5 path."""
    base_path = Path(base_dir) / name

    if name.endswith('.hdf5') or name.endswith('.h5'):
        return load_hdf5_dataset(name)

    fvecs_datasets = {
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

    hdf5_datasets = {
        "glove200": "glove200/glove-200-angular.hdf5",
    }

    if name in fvecs_datasets:
        files = fvecs_datasets[name]
        base = load_fvecs(base_path / files["base"])
        queries = load_fvecs(base_path / files["queries"])
        groundtruth = load_ivecs(base_path / files["groundtruth"])
        return {
            "base": base,
            "queries": queries,
            "groundtruth": groundtruth,
            "dim": base.shape[1],
        }
    elif name in hdf5_datasets:
        hdf5_path = Path(base_dir) / hdf5_datasets[name]
        return load_hdf5_dataset(str(hdf5_path))
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Supported: {list(fvecs_datasets.keys()) + list(hdf5_datasets.keys())}, "
            f"or pass a path to an HDF5 file."
        )
