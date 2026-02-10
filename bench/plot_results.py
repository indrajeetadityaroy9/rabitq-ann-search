#!/usr/bin/env python3
"""Plot ANN benchmark results: Recall-QPS Pareto curves, build time, memory.

Usage:
    python bench/plot_results.py bench/results/sift1m_results.json
    python bench/plot_results.py bench/results/*.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Color / marker scheme
# ---------------------------------------------------------------------------

ALGO_STYLE = {
    # CP-HNSW — warm colors
    "cphnsw-flat-1bit-M32":   {"color": "#d62728", "marker": "o",  "label": "CP-HNSW flat 1-bit"},
    "cphnsw-flat-2bit-M32":   {"color": "#ff7f0e", "marker": "s",  "label": "CP-HNSW flat 2-bit"},
    "cphnsw-flat-4bit-M32":   {"color": "#e377c2", "marker": "^",  "label": "CP-HNSW flat 4-bit"},
    "cphnsw-hnsw-1bit-M32":   {"color": "#2ca02c", "marker": "D",  "label": "CP-HNSW HNSW 1-bit"},
    # Competitors — cool colors
    "hnswlib-M16":            {"color": "#1f77b4", "marker": "v",  "label": "hnswlib M=16"},
    "hnswlib-M32":            {"color": "#17becf", "marker": "<",  "label": "hnswlib M=32"},
    "hnswlib-M64":            {"color": "#9467bd", "marker": ">",  "label": "hnswlib M=64"},
    "faiss-hnsw-M32":         {"color": "#8c564b", "marker": "p",  "label": "FAISS HNSW M=32"},
    "faiss-ivfflat-1024":     {"color": "#7f7f7f", "marker": "h",  "label": "FAISS IVF-Flat"},
    "faiss-ivfpq-1024":       {"color": "#bcbd22", "marker": "*",  "label": "FAISS IVF-PQ"},
}

def get_style(name):
    if name in ALGO_STYLE:
        return ALGO_STYLE[name]
    # Fallback
    return {"color": "#333333", "marker": "x", "label": name}


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_recall_qps(results, dataset_name, output_dir):
    """Recall@10 vs QPS Pareto curve."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for algo in results:
        name = algo["algorithm"]
        style = get_style(name)
        sweep = algo["sweep"]
        if not sweep:
            continue

        recalls = [p["recall_at_10"] for p in sweep]
        qps_vals = [p["qps"] for p in sweep]

        ax.plot(recalls, qps_vals,
                color=style["color"], marker=style["marker"],
                markersize=7, linewidth=1.5, label=style["label"])

    ax.set_xlabel("Recall@10", fontsize=13)
    ax.set_ylabel("QPS (queries/sec)", fontsize=13)
    ax.set_yscale("log")
    ax.set_title(f"Recall@10 vs QPS — {dataset_name}", fontsize=14)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.02)

    outpath = Path(output_dir) / f"{dataset_name}_recall_qps.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_build_time(results, dataset_name, output_dir):
    """Horizontal bar chart of build times."""
    names = []
    times = []
    colors = []
    for algo in results:
        if algo["build_time_s"] <= 0:
            continue
        style = get_style(algo["algorithm"])
        names.append(style["label"])
        times.append(algo["build_time_s"])
        colors.append(style["color"])

    if not names:
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.5 + 1)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, times, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Build Time (seconds)", fontsize=12)
    ax.set_title(f"Build Time — {dataset_name}", fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for i, t in enumerate(times):
        ax.text(t + max(times) * 0.01, i, f"{t:.1f}s", va="center", fontsize=9)

    outpath = Path(output_dir) / f"{dataset_name}_build_time.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_memory(results, dataset_name, output_dir):
    """Horizontal bar chart of memory usage."""
    names = []
    mem_vals = []
    colors = []
    for algo in results:
        if algo["memory_mb"] <= 0:
            continue
        style = get_style(algo["algorithm"])
        names.append(style["label"])
        mem_vals.append(algo["memory_mb"])
        colors.append(style["color"])

    if not names:
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.5 + 1)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, mem_vals, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Memory (MB)", fontsize=12)
    ax.set_title(f"Memory Usage — {dataset_name}", fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for i, m in enumerate(mem_vals):
        ax.text(m + max(mem_vals) * 0.01, i, f"{m:.0f} MB", va="center", fontsize=9)

    outpath = Path(output_dir) / f"{dataset_name}_memory.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(filepath: str):
    """Generate all plots for one result file."""
    with open(filepath) as f:
        data = json.load(f)

    meta = data["metadata"]
    results = data["results"]
    dataset_name = meta["dataset"].replace(" ", "-")
    output_dir = Path(filepath).parent

    print(f"\nPlotting {dataset_name} ({len(results)} algorithms)")

    plot_recall_qps(results, dataset_name, output_dir)
    plot_build_time(results, dataset_name, output_dir)
    plot_memory(results, dataset_name, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot ANN benchmark results")
    parser.add_argument("files", nargs="+", help="JSON result files")
    args = parser.parse_args()

    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"WARNING: {filepath} not found, skipping")
            continue
        process_file(filepath)

    print("\nDone.")


if __name__ == "__main__":
    main()
