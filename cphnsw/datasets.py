"""Dataset loaders and download helpers."""

import tarfile
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


DEFAULT_QUERY_SPLIT_SEED = 42


def load_fvecs(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    d = data[:1].view(np.int32)[0]
    return data.reshape(-1, d + 1)[:, 1:].copy()


def load_ivecs(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    k = int(data[0])
    return data.reshape(-1, k + 1)[:, 1:].copy()


def _compute_groundtruth(base: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray:
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base.astype(np.float32))
    _, ids = index.search(queries.astype(np.float32), k)
    return ids.astype(np.int32)


def _download_tar_dataset(repo_id: str, filename: str, dest: Path):
    local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with tarfile.open(local, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                member.name = Path(member.name).name
                tar.extract(member, dest)


def _split_queries(data: np.ndarray, n_queries: int = 1000):
    rng = np.random.default_rng(DEFAULT_QUERY_SPLIT_SEED)
    query_idx = rng.choice(len(data), size=n_queries, replace=False)
    mask = np.ones(len(data), dtype=bool)
    mask[query_idx] = False
    return data[mask], data[query_idx]


def download_sift1m(dest: str = "data/sift1m/"):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _download_tar_dataset("qbo-odp/sift1m", "sift.tar.gz", dest)


def download_gist1m(dest: str = "data/gist1m/"):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _download_tar_dataset("fzliu/gist1m", "gist.tar.gz", dest)


def download_openai1536(dest: str = "data/openai1536/"):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    repo_id = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M"

    parquet_path = hf_hub_download(repo_id=repo_id, filename="data/train-00000-of-00001.parquet",
                                   repo_type="dataset")
    df = pd.read_parquet(parquet_path)
    embeddings = np.stack(df["text-embedding-3-large-1536-embedding"].values).astype(np.float32)

    base, queries = _split_queries(embeddings)
    gt = _compute_groundtruth(base, queries, k=100)

    np.save(dest / "base.npy", base)
    np.save(dest / "queries.npy", queries)
    np.save(dest / "groundtruth.npy", gt)


def download_msmarco10m(dest: str = "data/msmarco10m/"):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    repo_id = "Cohere/msmarco-v2.1-embed-english-v3"
    target_n = 10_000_000

    api = HfApi()
    shard_files = sorted(
        si.rfilename for si in api.list_repo_tree(
            repo_id, path_in_repo="passages_npy", repo_type="dataset"
        ) if si.rfilename.endswith(".npy")
    )

    chunks = []
    total = 0
    for fname in shard_files:
        local = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
        chunk = np.load(local).astype(np.float32)
        chunks.append(chunk)
        total += len(chunk)
        if total >= target_n:
            break

    base_full = np.concatenate(chunks)[:target_n]
    base, queries = _split_queries(base_full)
    gt = _compute_groundtruth(base, queries, k=100)

    np.save(dest / "base.npy", base)
    np.save(dest / "queries.npy", queries)
    np.save(dest / "groundtruth.npy", gt)


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
