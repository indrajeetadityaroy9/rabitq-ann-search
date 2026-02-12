"""GPU helpers for ANN evaluation."""

import numpy as np


def _check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available")


def gpu_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize vectors on GPU."""
    import torch
    _check_cuda()
    t = torch.from_numpy(X).cuda()
    t = t / t.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return t.cpu().numpy()


def gpu_ground_truth(base: np.ndarray, queries: np.ndarray, k: int,
                     chunk_size: int = 512) -> np.ndarray:
    """Compute exact L2 kNN IDs on GPU."""
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
    """Compute pairwise L2 distances on GPU."""
    import torch
    _check_cuda()
    Y_t = torch.from_numpy(Y).cuda()

    chunks = []
    for i in range(0, len(X), chunk_size):
        X_chunk = torch.from_numpy(X[i:i + chunk_size]).cuda()
        d = torch.cdist(X_chunk, Y_t)
        chunks.append(d.cpu())

    return torch.cat(chunks, dim=0).numpy()
