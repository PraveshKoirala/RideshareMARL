"""Plot rate and commission for U and L on both routes from metrics.jsonl."""

import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--responsive", default="out_responsive/metrics.jsonl")
parser.add_argument("--lagging", default="out_lagging/metrics.jsonl")
parser.add_argument("--out-dir", default="out_plots")
parser.add_argument("--alpha", type=float, default=0.1, help="EMA smoothing factor")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

regimes = [
    ("Responsive", args.responsive),
    ("Lagging", args.lagging),
]

plots = [
    ("U_route_0_1", "RU_0_1", "CU_0_1", "U: Route 0→1"),
    ("U_route_1_0", "RU_1_0", "CU_1_0", "U: Route 1→0"),
    ("L_route_0_1", "RL_0_1", "CL_0_1", "L: Route 0→1"),
    ("L_route_1_0", "RL_1_0", "CL_1_0", "L: Route 1→0"),
]

for regime_name, metrics_path in regimes:
    records = [json.loads(line) for line in open(metrics_path)]
    df = pd.DataFrame(records)

    for fname, rate_col, comm_col, title in plots:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[rate_col].ewm(alpha=args.alpha).mean(), label="Rate")
        ax.plot(df[comm_col].ewm(alpha=args.alpha).mean(), label="Commission")
        ax.set_title(f"{title} ({regime_name})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("$/mile")
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"{fname}_{regime_name.lower()}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

print(f"Saved 8 plots to {args.out_dir}/")
