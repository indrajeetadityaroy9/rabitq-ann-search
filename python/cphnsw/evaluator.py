"""Evaluation framework for ANN index benchmarking."""

import csv
import time
from pathlib import Path

import numpy as np

from .metrics import recall_at_k, qps


class Evaluator:
    """Runs recall-vs-qps evaluation sweeps on an ANN index."""

    def __init__(self, index, dataset: dict, config: dict):
        """
        Args:
            index: A cphnsw.Index instance (must be finalized).
            dataset: Dict with keys: queries, groundtruth, dim.
            config: Dict with keys: k, ef_values, warmup_queries.
        """
        self.index = index
        self.queries = dataset["queries"]
        self.groundtruth = dataset["groundtruth"]
        self.k = config.get("k", 10)
        self.ef_values = config.get("ef_values", [10, 20, 40, 80, 100, 200, 400])
        self.warmup = config.get("warmup_queries", 100)
        self.results = []

    def run(self) -> list:
        """Run evaluation sweep over ef values.

        Returns:
            List of dicts with keys: ef, recall, qps, p50_us, p99_us.
        """
        n_queries = len(self.queries)
        self.results = []

        for ef in self.ef_values:
            # Warmup
            for i in range(min(self.warmup, n_queries)):
                self.index.search(self.queries[i], k=self.k, ef=ef)

            # Timed evaluation
            latencies = []
            all_ids = []
            for i in range(n_queries):
                t0 = time.perf_counter()
                ids, dists = self.index.search(self.queries[i], k=self.k, ef=ef)
                latencies.append(time.perf_counter() - t0)
                all_ids.append(ids)

            latencies = np.array(latencies)
            recall_sum = 0.0
            for i in range(n_queries):
                recall_sum += recall_at_k(all_ids[i], self.groundtruth[i], self.k)
            mean_recall = recall_sum / n_queries

            self.results.append({
                "ef": ef,
                "recall": mean_recall,
                "qps": qps(latencies),
                "p50_us": np.percentile(latencies, 50) * 1e6,
                "p99_us": np.percentile(latencies, 99) * 1e6,
            })

        return self.results

    def save(self, path: str):
        """Save results to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ef", "recall", "qps", "p50_us", "p99_us"])
            writer.writeheader()
            writer.writerows(self.results)
