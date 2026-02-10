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


def download_sift1m(dest: str = "data/sift1m/"):
    """Download SIFT-1M dataset from the standard mirror.

    Downloads and extracts sift_base.fvecs, sift_query.fvecs,
    sift_groundtruth.ivecs into *dest*.
    """
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "sift_base.fvecs").exists():
        print(f"SIFT-1M already exists at {dest}")
        return

    tarball = dest / "sift.tar.gz"
    print(f"Downloading SIFT-1M to {tarball} ...")
    urllib.request.urlretrieve(url, tarball)

    print("Extracting ...")
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                member.name = Path(member.name).name  # flatten
                tar.extract(member, dest)

    tarball.unlink()
    print(f"SIFT-1M ready at {dest}")


def load_dataset(name: str, base_dir: str = "data/") -> dict:
    """Load a standard ANN benchmark dataset.

    Supported names: sift1m, gist1m.

    Returns:
        dict with keys: base, queries, groundtruth, dim.
    """
    base_path = Path(base_dir) / name

    datasets = {
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

    if name not in datasets:
        raise ValueError(f"Unknown dataset '{name}'. Supported: {list(datasets.keys())}")

    files = datasets[name]
    base = load_fvecs(base_path / files["base"])
    queries = load_fvecs(base_path / files["queries"])
    groundtruth = load_ivecs(base_path / files["groundtruth"])

    return {
        "base": base,
        "queries": queries,
        "groundtruth": groundtruth,
        "dim": base.shape[1],
    }
