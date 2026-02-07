"""Plotting utilities for ANN benchmark results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(csv_path: str) -> dict:
    """Load evaluation results from CSV.

    Returns:
        Dict with arrays: ef, recall, qps, p50_us, p99_us.
    """
    import csv
    data = {"ef": [], "recall": [], "qps": [], "p50_us": [], "p99_us": []}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))
    return {k: np.array(v) for k, v in data.items()}


def plot_recall_qps(csv_path: str, output_path: str, label: str = "CP-HNSW"):
    """Plot recall@k vs QPS curve."""
    data = load_results(csv_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["recall"], data["qps"], "o-", label=label, linewidth=2, markersize=6)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries/second")
    ax.set_yscale("log")
    ax.set_title("Recall vs Throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_construction(csv_path: str, output_path: str):
    """Plot construction throughput (build time vs dataset size)."""
    data = load_results(csv_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["ef"], data["qps"], "s-", color="tab:orange", linewidth=2, markersize=6)
    ax.set_xlabel("ef_construction")
    ax.set_ylabel("Build throughput (vectors/s)")
    ax.set_title("Construction Performance")
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_memory(csv_path: str, output_path: str):
    """Plot memory usage vs recall tradeoff."""
    data = load_results(csv_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["recall"], data["ef"], "d-", color="tab:green", linewidth=2, markersize=6)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Search ef (proxy for memory pressure)")
    ax.set_title("Memory / Accuracy Tradeoff")
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recall_qps_multi(csv_paths: dict, output_path: str):
    """Plot multiple recall-QPS curves for comparison.

    Args:
        csv_paths: Dict mapping label -> csv_path.
        output_path: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, path in csv_paths.items():
        data = load_results(path)
        ax.plot(data["recall"], data["qps"], "o-", label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries/second")
    ax.set_yscale("log")
    ax.set_title("Recall vs Throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
