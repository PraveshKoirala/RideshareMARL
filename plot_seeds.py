#!/usr/bin/env python3
"""Plot mean ± 95% CI from multi-seed traces, aggregated per epoch."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from configparser import ConfigParser

parser = argparse.ArgumentParser()
parser.add_argument("--traces-dir", default="out_seeds")
parser.add_argument("--out-dir", default="out_plots_ci")
parser.add_argument("--alpha", type=float, default=0.02, help="EMA alpha for smoothing")
args = parser.parse_args()

traces_dir = Path(args.traces_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# Read traj_length from config
cfg = ConfigParser()
cfg.read("config.ini")
traj_length = int(eval(cfg.get("ppo", "traj_length")))


def load_traces(regime):
    """Load all seed traces for a regime, return dict of arrays (n_seeds, n_epochs)."""
    files = sorted(traces_dir.glob(f"trace_{regime}_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"No traces found for {regime} in {traces_dir}")
    all_data = {}
    for f in files:
        data = np.load(f)
        for key in data.files:
            # Average per-step values within each epoch -> per-epoch values
            vals = data[key]
            n_epochs = len(vals) // traj_length
            epoch_means = vals[:n_epochs * traj_length].reshape(n_epochs, traj_length).mean(axis=1)
            all_data.setdefault(key, []).append(epoch_means)
    return {k: np.array(v) for k, v in all_data.items()}


def ema(arr, alpha):
    """EMA along axis=1 (per-seed smoothing before aggregation)."""
    out = np.zeros_like(arr)
    out[:, 0] = arr[:, 0]
    for i in range(1, arr.shape[1]):
        out[:, i] = alpha * arr[:, i] + (1 - alpha) * out[:, i - 1]
    return out


def plot_with_ci(ax, epochs, data, label, color):
    """Plot individual seed traces (low opacity) + mean (solid)."""
    smoothed = ema(data, args.alpha)
    for i in range(smoothed.shape[0]):
        ax.plot(epochs, smoothed[i], color=color, alpha=0.25, linewidth=0.5)
    mean = smoothed.mean(axis=0)
    ax.plot(epochs, mean, color=color, label=label, linewidth=2)


plots = [
    ("U_route_0_1", "RU_0_1", "CU_0_1", "U: Route 0→1"),
    ("U_route_1_0", "RU_1_0", "CU_1_0", "U: Route 1→0"),
    ("L_route_0_1", "RL_0_1", "CL_0_1", "L: Route 0→1"),
    ("L_route_1_0", "RL_1_0", "CL_1_0", "L: Route 1→0"),
]

for regime in ["responsive", "lagging"]:
    data = load_traces(regime)
    n_epochs = data["RU_0_1"].shape[1]
    epochs = np.arange(n_epochs)

    for fname, rate_key, comm_key, title in plots:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_with_ci(ax, epochs, data[rate_key], "Rate", "#DF672A")
        plot_with_ci(ax, epochs, data[comm_key], "Commission", "#338DD8")
        ax.set_title(f"{title} ({regime.capitalize()})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("$/mile")
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname}_{regime}.png", dpi=150)
        plt.close(fig)

print(f"Saved {2 * len(plots)} plots to {out_dir}/")
