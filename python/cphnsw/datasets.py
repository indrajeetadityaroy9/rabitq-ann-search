"""Dataset loading utilities for standard ANN benchmark formats."""

import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


def load_fvecs(path: str) -> np.ndarray:
    """Load vectors from .fvecs format.

    Format: each record is [int32 dim][dim x float32 values].

    Returns:
        (n, dim) float32 array.
    """
    data = np.fromfile(path, dtype=np.float32)
    d = data[:1].view(np.int32)[0]
    return data.reshape(-1, d + 1)[:, 1:].copy()


def load_ivecs(path: str) -> np.ndarray:
    """Load integer vectors from .ivecs format.

    Format: each record is [int32 k][k x int32 values].

    Returns:
        (n, k) int32 array.
    """
    data = np.fromfile(path, dtype=np.int32)
    k = int(data[0])
    return data.reshape(-1, k + 1)[:, 1:].copy()


def load_hdf5_dataset(path: str) -> dict:
    """Load ANN-benchmarks HDF5 format dataset.

    Expected keys in file: 'train', 'test', 'neighbors', 'distances'.

    Returns:
        dict with keys: base, queries, groundtruth, dim.
    """
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
    """Download SIFT-1M dataset."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "sift_base.fvecs").exists():
        print(f"SIFT-1M already exists at {dest}")
        return

    tarball = dest / "sift.tar.gz"
    print(f"Downloading SIFT-1M to {tarball} ...")
    urllib.request.urlretrieve(SIFT1M_URL, tarball)

    print("Extracting ...")
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                member.name = Path(member.name).name  # flatten
                tar.extract(member, dest)

    tarball.unlink()
    print(f"SIFT-1M ready at {dest}")


def download_glove200(dest: str = "data/glove200/"):
    """Download GloVe-200 dataset in HDF5 format."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    hdf5_path = dest / "glove-200-angular.hdf5"
    if hdf5_path.exists():
        print(f"GloVe-200 already exists at {hdf5_path}")
        return

    print(f"Downloading GloVe-200 to {hdf5_path} ...")
    urllib.request.urlretrieve(GLOVE200_URL, hdf5_path)
    print(f"GloVe-200 ready at {hdf5_path}")


def load_dataset(name: str, base_dir: str = "data/") -> dict:
    """Load a standard ANN benchmark dataset.

    Supported names: sift1m, gist1m, glove200, or a path to an HDF5 file.

    Returns:
        dict with keys: base, queries, groundtruth, dim.
    """
    base_path = Path(base_dir) / name

    # Check if it's an HDF5 file path
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
