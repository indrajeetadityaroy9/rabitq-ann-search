"""CP-HNSW Python package."""

from cphnsw.datasets import load_dataset, load_fvecs, load_ivecs
from cphnsw.metrics import qps, recall_at_k

__all__ = [
    "Index",
    "HNSWIndex",
    "load_fvecs",
    "load_ivecs",
    "load_dataset",
    "recall_at_k",
    "qps",
]


def __getattr__(name):
    if name == "Index":
        from cphnsw._core import Index

        return Index
    if name == "HNSWIndex":
        from cphnsw._core import HNSWIndex

        return HNSWIndex
    raise AttributeError(f"module 'cphnsw' has no attribute '{name}'")
