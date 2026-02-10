"""GPU-accelerated helpers for CP-HNSW using PyTorch CUDA."""

import numpy as np


def _check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available")


def gpu_normalize(X: np.ndarray) -> np.ndarray:
    """Batch L2-normalize vectors on GPU.

    Args:
        X: (n, d) float32 array.

    Returns:
        (n, d) float32 array with unit L2 norms.
    """
    import torch
    _check_cuda()
    t = torch.from_numpy(X).cuda()
    t = t / t.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return t.cpu().numpy()


def gpu_ground_truth(base: np.ndarray, queries: np.ndarray, k: int,
                     chunk_size: int = 512) -> np.ndarray:
    """Brute-force kNN on GPU via PyTorch.

    Computes exact L2 nearest neighbors. Handles 1M+ vectors on H100 (80 GB).

    Args:
        base: (n, d) float32 base vectors.
        queries: (nq, d) float32 query vectors.
        k: Number of neighbors.
        chunk_size: Query batch size to avoid OOM.

    Returns:
        (nq, k) int64 array of neighbor IDs.
    """
    import torch
    _check_cuda()
    base_t = torch.from_numpy(base).cuda()
    query_t = torch.from_numpy(queries).cuda()

    all_ids = []
    for i in range(0, len(queries), chunk_size):
        q = query_t[i:i + chunk_size]
        dists = torch.cdist(q, base_t)
        _, ids = dists.topk(k, largest=False)
        all_ids.append(ids.cpu())

    return torch.cat(all_ids, dim=0).numpy().astype(np.int64)


def gpu_pairwise_distances(X: np.ndarray, Y: np.ndarray,
                           chunk_size: int = 1024) -> np.ndarray:
    """Compute pairwise L2 distances on GPU.

    Args:
        X: (n, d) float32 array.
        Y: (m, d) float32 array.
        chunk_size: Row batch size for X to avoid OOM.

    Returns:
        (n, m) float32 distance matrix.
    """
    import torch
    _check_cuda()
    Y_t = torch.from_numpy(Y).cuda()

    chunks = []
    for i in range(0, len(X), chunk_size):
        X_chunk = torch.from_numpy(X[i:i + chunk_size]).cuda()
        d = torch.cdist(X_chunk, Y_t)
        chunks.append(d.cpu())

    return torch.cat(chunks, dim=0).numpy()
