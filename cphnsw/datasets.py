"""Dataset loaders and download helpers."""

import json
import tarfile
from pathlib import Path

import faiss
import numpy as np
from huggingface_hub import hf_hub_download


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


def _compute_groundtruth(base: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray:
    """Compute exact k-NN ground truth using FAISS brute force."""
    _emit("groundtruth_compute_start", n_base=len(base), n_queries=len(queries), k=k)
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base.astype(np.float32))
    _, ids = index.search(queries.astype(np.float32), k)
    _emit("groundtruth_compute_done", n_base=len(base), n_queries=len(queries), k=k)
    return ids.astype(np.int32)


# --- Download helpers ---

def download_sift1m(dest: str = "data/sift1m/"):
    """Download SIFT-1M if missing."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "sift_base.fvecs").exists():
        _emit("dataset_cache_hit", dataset="sift1m", path=str(dest))
        return

    _emit("dataset_download_start", dataset="sift1m", repo="qbo-odp/sift1m", path=str(dest))
    local = hf_hub_download(repo_id="qbo-odp/sift1m", filename="sift.tar.gz", repo_type="dataset")
    with tarfile.open(local, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                member.name = Path(member.name).name
                tar.extract(member, dest)
    _emit("dataset_ready", dataset="sift1m", path=str(dest))


def download_gist1m(dest: str = "data/gist1m/"):
    """Download GIST-1M if missing."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "gist_base.fvecs").exists():
        _emit("dataset_cache_hit", dataset="gist1m", path=str(dest))
        return

    _emit("dataset_download_start", dataset="gist1m", repo="fzliu/gist1m", path=str(dest))
    local = hf_hub_download(repo_id="fzliu/gist1m", filename="gist.tar.gz", repo_type="dataset")
    with tarfile.open(local, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                member.name = Path(member.name).name
                tar.extract(member, dest)
    _emit("dataset_ready", dataset="gist1m", path=str(dest))


def download_openai1536(dest: str = "data/openai1536/"):
    """Download OpenAI text-embedding-3-large 1536D (999K vectors) from HuggingFace."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "base.npy").exists():
        _emit("dataset_cache_hit", dataset="openai1536", path=str(dest))
        return

    repo_id = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M"
    _emit("dataset_download_start", dataset="openai1536", repo=repo_id, path=str(dest))

    import pandas as pd
    parquet_path = hf_hub_download(repo_id=repo_id, filename="data/train-00000-of-00001.parquet",
                                   repo_type="dataset")
    df = pd.read_parquet(parquet_path)
    embeddings = np.stack(df["text-embedding-3-large-1536-embedding"].values).astype(np.float32)

    # Sample 1000 queries from base, remove from base
    rng = np.random.RandomState(42)
    query_idx = rng.choice(len(embeddings), size=1000, replace=False)
    mask = np.ones(len(embeddings), dtype=bool)
    mask[query_idx] = False
    queries = embeddings[query_idx]
    base = embeddings[mask]

    gt = _compute_groundtruth(base, queries, k=100)

    np.save(dest / "base.npy", base)
    np.save(dest / "queries.npy", queries)
    np.save(dest / "groundtruth.npy", gt)
    _emit("dataset_ready", dataset="openai1536", path=str(dest))


def download_msmarco10m(dest: str = "data/msmarco10m/"):
    """Download MSMARCO 10M subset (1024D) from Cohere HuggingFace."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "base.npy").exists():
        _emit("dataset_cache_hit", dataset="msmarco10m", path=str(dest))
        return

    from huggingface_hub import HfApi
    repo_id = "Cohere/msmarco-v2.1-embed-english-v3"
    target_n = 10_000_000
    _emit("dataset_download_start", dataset="msmarco10m", repo=repo_id, path=str(dest))

    # List npy shard files in passages_npy/
    api = HfApi()
    shard_files = sorted(
        si.rfilename for si in api.list_repo_tree(
            repo_id, path_in_repo="passages_npy", repo_type="dataset"
        ) if si.rfilename.endswith(".npy")
    )

    # Download shards until we accumulate 10M vectors
    chunks = []
    total = 0
    for fname in shard_files:
        _emit("dataset_download_shard", dataset="msmarco10m", shard=fname, accumulated=total)
        local = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
        chunk = np.load(local).astype(np.float32)
        chunks.append(chunk)
        total += len(chunk)
        if total >= target_n:
            break

    base_full = np.concatenate(chunks)[:target_n]

    # Sample 1000 queries from base, remove from base
    rng = np.random.RandomState(42)
    query_idx = rng.choice(len(base_full), size=1000, replace=False)
    mask = np.ones(len(base_full), dtype=bool)
    mask[query_idx] = False
    queries = base_full[query_idx]
    base = base_full[mask]

    gt = _compute_groundtruth(base, queries, k=100)

    np.save(dest / "base.npy", base)
    np.save(dest / "queries.npy", queries)
    np.save(dest / "groundtruth.npy", gt)
    _emit("dataset_ready", dataset="msmarco10m", path=str(dest))


# --- Dataset registry and loader ---

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
    """Load a named benchmark dataset."""
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
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {ALL_DATASETS}."
        )

    return {
        "base": base,
        "queries": queries,
        "groundtruth": groundtruth,
        "dim": base.shape[1],
    }
