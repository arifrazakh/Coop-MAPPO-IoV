"""Plot training metrics from ppo_episode_metrics.csv.

The CSV is produced by :class:`mamn_marl.callbacks.EpisodeCSVLogger`.

Examples
--------
python scripts/plot_csv.py --csv ppo_episode_metrics.csv
python scripts/plot_csv.py --csv /path/to/ppo_episode_metrics.csv --out outputs
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: str) -> Dict[str, np.ndarray]:
    """Read the episode CSV into a column dict."""
    cols: Dict[str, List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    cols.setdefault(k, []).append(float(v))
                except ValueError:
                    # ignore non-numeric columns
                    pass

    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def save_line_plot(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: str) -> None:
    plt.figure(figsize=(9, 4.5), dpi=160)
    plt.plot(x, y, linewidth=1.4)
    plt.grid(True, which="major", linestyle="-", alpha=0.35)
    plt.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.minorticks_on()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="ppo_episode_metrics.csv", help="CSV file to read")
    ap.add_argument("--out", type=str, default="outputs", help="Output directory")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    os.makedirs(args.out, exist_ok=True)
    data = read_csv(args.csv)

    # Choose an x-axis: episode_id if present; else row index
    if "episode_id" in data:
        x = data["episode_id"]
    else:
        n = len(next(iter(data.values())))
        x = np.arange(n)

    # Common metrics (only plot if present)
    candidates = [
        ("episode_return", "Episode Return (sum of global_reward)", "Episode", "Return"),
        ("total_throughput_Mbps", "Total Served Throughput", "Episode", "Mbps"),
        ("avg_latency_ms", "Average End-to-End Latency", "Episode", "ms"),
        ("blocking_rate", "Blocking Rate", "Episode", "Fraction"),
        ("jain_fairness_bs", "Jain Fairness (BS-level)", "Episode", "Index"),
        ("deadline_satisfaction", "Deadline Satisfaction", "Episode", "Fraction"),
    ]

    for key, title, xlabel, ylabel in candidates:
        if key not in data:
            continue
        out_path = os.path.join(args.out, f"{key}.png")
        save_line_plot(x[: len(data[key])], data[key], title, xlabel, ylabel, out_path)

    print(f"Saved plots to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
