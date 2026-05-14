# Algorithmic collusion in double-sided markets: A rideshare example

The paper related to this repo has been submitted to ITSC 2024 and has been accepted.
The PPO implementation used has been taken from https://github.com/seolhokim/Mujoco-Pytorch

## Dependencies

```
pytorch
pettingzoo
numpy
matplotlib
pandas
```

Install: `pip install torch pettingzoo numpy matplotlib pandas`

## Quick Start

### Single run (one regime)

```bash
# Responsive market (drivers adjust freely)
RIDESHARE_LOG_DIR=./out_responsive python3 run-custom.py --epochs 500 --a_d 1.0

# Lagging market (drivers adjust slowly, ±0.05 per step)
RIDESHARE_LOG_DIR=./out_lagging python3 run-custom.py --epochs 500 --a_d 0.05
```

### Multi-seed experiment (20 seeds × 2 regimes)

```bash
python3 run_seeds.py
```

This runs 40 jobs in parallel on CPU and saves per-step traces to `out_seeds/`.

### Plotting

```bash
# Single-run plots (requires out_responsive/ and out_lagging/)
python3 plot_results.py

# Multi-seed plots with individual traces + mean (requires out_seeds/)
python3 plot_seeds.py
```

Plots are saved to `out_plots/` and `out_plots_ci/` respectively.

## Configuration

PPO hyperparameters are in `config.ini`. OD matrix and cost matrix can be directly changed in `run-custom.py`.

| Parameter | Value | Description |
|-----------|-------|-------------|
| hidden_dim | 1024 | Network width |
| traj_length | 2048 | Steps per epoch |
| max_rate | 20 | Maximum $/mile |
| g | 5 | Gas cost (commission floor) |
| rp | 10 | Public transit rate |
| lbd | 2 | Wait-cost multiplier λ |
| a_d | 1.0 / 0.05 | Driver step (responsive / lagging) |

## File Structure

| File | Purpose |
|------|---------|
| `run-custom.py` | Main training script (single run) |
| `run_seeds.py` | Multi-seed parallel runner |
| `plot_results.py` | Plot single-run results |
| `plot_seeds.py` | Plot multi-seed results with traces |
| `config.ini` | PPO hyperparameters |
| `customPPO.py` | PPO implementation |
| `rideshare/env_multi.py` | Multi-node rideshare environment |
| `utils/file_logger.py` | Drop-in wandb replacement (writes JSONL) |
| `utils/utils.py` | Replay buffer, utilities |

## Citation

Pravesh Koirala, Forrest Laine. "Algorithmic Collusion in Double-Sided Markets: A Rideshare Example." *2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC)*, pp. 3445–3452, IEEE, 2024.
