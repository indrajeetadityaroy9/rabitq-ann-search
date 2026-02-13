"""Generate benchmark plots from result JSON files."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


ALGO_STYLE = {
    "cphnsw-flat-1bit":  {"color": "#d62728", "marker": "o",  "label": "CP-HNSW flat 1-bit"},
    "cphnsw-flat-2bit":  {"color": "#ff7f0e", "marker": "s",  "label": "CP-HNSW flat 2-bit"},
    "cphnsw-flat-4bit":  {"color": "#e377c2", "marker": "^",  "label": "CP-HNSW flat 4-bit"},
    "cphnsw-hnsw-1bit":  {"color": "#2ca02c", "marker": "D",  "label": "CP-HNSW HNSW 1-bit"},
    "hnswlib-M16":       {"color": "#1f77b4", "marker": "v",  "label": "hnswlib M=16"},
    "hnswlib-M32":       {"color": "#17becf", "marker": "<",  "label": "hnswlib M=32"},
    "hnswlib-M64":       {"color": "#9467bd", "marker": ">",  "label": "hnswlib M=64"},
    "faiss-hnsw-M32":    {"color": "#8c564b", "marker": "p",  "label": "FAISS HNSW M=32"},
    "faiss-ivfpq":       {"color": "#bcbd22", "marker": "*",  "label": "FAISS IVF-PQ"},
    "faiss-ivfopq":      {"color": "#aec7e8", "marker": "P",  "label": "FAISS IVF-OPQ"},
}

def get_style(name):
    if name in ALGO_STYLE:
        return ALGO_STYLE[name]
    return {"color": "#333333", "marker": "x", "label": name}


def plot_recall_qps(results, dataset_name, output_dir, recall_key="recall_at_10", label_suffix="10"):
    """Plot Recall@k vs QPS."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for algo in results:
        name = algo["algorithm"]
        style = get_style(name)
        sweep = algo["sweep"]
        if not sweep:
            continue

        recalls = [p.get(recall_key, 0) for p in sweep]
        qps_vals = [p["qps"] for p in sweep]

        ax.plot(recalls, qps_vals,
                color=style["color"], marker=style["marker"],
                markersize=7, linewidth=1.5, label=style["label"])

    ax.set_xlabel(f"Recall@{label_suffix}", fontsize=13)
    ax.set_ylabel("QPS (queries/sec)", fontsize=13)
    ax.set_yscale("log")
    ax.set_title(f"Recall@{label_suffix} vs QPS — {dataset_name}", fontsize=14)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.02)

    outpath = Path(output_dir) / f"{dataset_name}_recall{label_suffix}_qps.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    emit("plot_written", dataset=dataset_name, plot=f"recall{label_suffix}_qps", path=str(outpath))


def plot_adr_qps(results, dataset_name, output_dir):
    """Plot ADR vs QPS."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for algo in results:
        name = algo["algorithm"]
        style = get_style(name)
        sweep = algo["sweep"]
        if not sweep or "adr" not in sweep[0]:
            continue

        adrs = [p["adr"] for p in sweep]
        qps_vals = [p["qps"] for p in sweep]

        ax.plot(adrs, qps_vals,
                color=style["color"], marker=style["marker"],
                markersize=7, linewidth=1.5, label=style["label"])

    ax.set_xlabel("Average Distance Ratio (ADR)", fontsize=13)
    ax.set_ylabel("QPS (queries/sec)", fontsize=13)
    ax.set_yscale("log")
    ax.set_title(f"ADR vs QPS — {dataset_name}", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    outpath = Path(output_dir) / f"{dataset_name}_adr_qps.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    emit("plot_written", dataset=dataset_name, plot="adr_qps", path=str(outpath))


def plot_build_time(results, dataset_name, output_dir):
    """Plot build-time bars."""
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

    for i, t in enumerate(times):
        ax.text(t + max(times) * 0.01, i, f"{t:.1f}s", va="center", fontsize=9)

    outpath = Path(output_dir) / f"{dataset_name}_build_time.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    emit("plot_written", dataset=dataset_name, plot="build_time", path=str(outpath))


def plot_memory(results, dataset_name, output_dir):
    """Plot memory bars."""
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

    for i, m in enumerate(mem_vals):
        ax.text(m + max(mem_vals) * 0.01, i, f"{m:.0f} MB", va="center", fontsize=9)

    outpath = Path(output_dir) / f"{dataset_name}_memory.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    emit("plot_written", dataset=dataset_name, plot="memory", path=str(outpath))


def process_file(filepath: str):
    """Generate all plots for one result file."""
    with open(filepath) as f:
        data = json.load(f)

    meta = data["metadata"]
    results = data["results"]
    dataset_name = meta["dataset"].replace(" ", "-")
    output_dir = Path(filepath).parent

    emit("plot_dataset_start", dataset=dataset_name, n_algorithms=len(results), source=str(filepath))

    plot_recall_qps(results, dataset_name, output_dir, "recall_at_10", "10")
    plot_recall_qps(results, dataset_name, output_dir, "recall_at_100", "100")
    plot_adr_qps(results, dataset_name, output_dir)
    plot_build_time(results, dataset_name, output_dir)
    plot_memory(results, dataset_name, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot ANN benchmark results")
    parser.add_argument("files", nargs="+", help="JSON result files")
    args = parser.parse_args()

    for filepath in args.files:
        process_file(filepath)


if __name__ == "__main__":
    main()
