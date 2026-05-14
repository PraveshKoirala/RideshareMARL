#!/usr/bin/env python3
"""Run 20 seeds x 2 regimes in parallel, record per-step data, plot with CI."""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
from copy import copy
from configparser import ConfigParser
from multiprocessing import get_context
from pathlib import Path

from customPPO import PPO
from rideshare.env_multi import MultiRideshareEnv
from utils.utils import make_transition, Dict, RunningMeanStd

N_SEEDS = 20
EPOCHS = 500
A_D_RESPONSIVE = 1.0
A_D_LAGGING = 0.05
OUT = Path("out_seeds")


def make_env_kwargs(a_d, traj_length):
    OD = np.array([[0., 0.9], [0.2, 0.]])
    C = np.array([[0., 5.], [2., 0.]])
    return {
        "OD": OD, "C": C,
        "init_passenger_distribution": np.array([2000., 3000.]),
        "init_driver_distribution": np.array([500., 1000.]),
        "max_rate": 20., "max_timestep": 8 * 864000,
        "lbd": 2., "rp": 10., "g": 5.,
        "num_D_samples": 10, "a_d": a_d, "p_d": 1.,
        "alpha": 1e-2, "mode": "interp",
        "traj_length": traj_length,
    }


def train_one(seed, a_d, agent_args, device="cpu"):
    """Train one run, return per-step rates/commissions as arrays."""
    traj_length = agent_args.traj_length
    env_kwargs = make_env_kwargs(a_d, traj_length)
    env = MultiRideshareEnv(**env_kwargs)
    N = env.N
    g = env.g
    max_rate = env.max_rate

    action_shape = env.action_space(None).shape
    action_dim = action_shape[0] * action_shape[1]
    state_dim = env.observation_space(None).shape[0]

    # Two PPO agents
    agent_U = PPO(None, device, state_dim, action_dim, agent_args)
    agent_L = PPO(None, device, state_dim, action_dim, agent_args)

    total_steps = EPOCHS * traj_length
    # Record per-step: RU, CU, RL, CL for each edge, plus profits
    # For 2-node: edges are (0,1) and (1,0)
    trace = {
        "RU_0_1": np.zeros(total_steps), "CU_0_1": np.zeros(total_steps),
        "RU_1_0": np.zeros(total_steps), "CU_1_0": np.zeros(total_steps),
        "RL_0_1": np.zeros(total_steps), "CL_0_1": np.zeros(total_steps),
        "RL_1_0": np.zeros(total_steps), "CL_1_0": np.zeros(total_steps),
        "profit_U": np.zeros(total_steps), "profit_L": np.zeros(total_steps),
    }

    torch.manual_seed(seed)
    np.random.seed(seed)
    observations, _ = env.reset(seed=seed)
    state = observations["U"]

    step_idx = 0
    for epoch in range(EPOCHS):
        for t in range(traj_length):
            # Sample actions
            mu_U, sigma_U = agent_U.get_action(torch.from_numpy(state).float().to(device))
            dist_U = torch.distributions.Normal(mu_U, sigma_U[0])
            action_U_t = dist_U.sample()
            log_prob_U = dist_U.log_prob(action_U_t).sum(-1, keepdim=True)

            mu_L, sigma_L = agent_L.get_action(torch.from_numpy(state).float().to(device))
            dist_L = torch.distributions.Normal(mu_L, sigma_L[0])
            action_L_t = dist_L.sample()
            log_prob_L = dist_L.log_prob(action_L_t).sum(-1, keepdim=True)

            action_U_np = action_U_t.cpu().detach().numpy().reshape(action_shape)
            action_L_np = action_L_t.cpu().detach().numpy().reshape(action_shape)

            # Compute rates for recording
            interp = lambda r: r * (max_rate - g) + g
            RU = np.clip(interp(action_U_np[:N, :]), g, max_rate)
            CU = np.clip(interp(action_U_np[N:, :]), g, max_rate)
            RL = np.clip(interp(action_L_np[:N, :]), g, max_rate)
            CL = np.clip(interp(action_L_np[N:, :]), g, max_rate)

            trace["RU_0_1"][step_idx] = RU[0, 1]
            trace["CU_0_1"][step_idx] = CU[0, 1]
            trace["RU_1_0"][step_idx] = RU[1, 0]
            trace["CU_1_0"][step_idx] = CU[1, 0]
            trace["RL_0_1"][step_idx] = RL[0, 1]
            trace["CL_0_1"][step_idx] = CL[0, 1]
            trace["RL_1_0"][step_idx] = RL[1, 0]
            trace["CL_1_0"][step_idx] = CL[1, 0]

            # Step env
            actions = {"U": action_U_np, "L": action_L_np}
            done = (t == traj_length - 1)
            observations, rewards, _, _, _ = env.step(actions, done)

            next_state = observations["U"]
            trace["profit_U"][step_idx] = rewards["U"]
            trace["profit_L"][step_idx] = rewards["L"]

            # Store transitions
            agent_U.put_data(make_transition(
                state, action_U_t.cpu().detach().numpy(),
                np.array([rewards["U"]]), next_state,
                np.array([done]), log_prob_U.detach().cpu().numpy()))
            agent_L.put_data(make_transition(
                state, action_L_t.cpu().detach().numpy(),
                np.array([rewards["L"]]), next_state,
                np.array([done]), log_prob_L.detach().cpu().numpy()))

            state = next_state
            step_idx += 1

        agent_U.train_net(epoch)
        agent_L.train_net(epoch)
        observations, _ = env.reset(seed=None)
        state = observations["U"]

    return trace


def run_job(args):
    seed, regime, a_d = args
    parser = ConfigParser()
    parser.read("config.ini")
    agent_args = Dict(parser, "ppo")
    print(f"  START {regime} seed={seed}", flush=True)
    trace = train_one(seed, a_d, agent_args, device="cpu")
    out_path = OUT / f"trace_{regime}_seed{seed}.npz"
    np.savez_compressed(out_path, **trace)
    print(f"  DONE  {regime} seed={seed}", flush=True)
    return str(out_path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in range(N_SEEDS):
        jobs.append((seed, "responsive", A_D_RESPONSIVE))
        jobs.append((seed + 100, "lagging", A_D_LAGGING))

    print(f"Launching {len(jobs)} jobs ({N_SEEDS} seeds x 2 regimes)")
    print(f"Config: epochs={EPOCHS}, traj_length from config.ini")

    ctx = get_context("spawn")
    with ctx.Pool(processes=min(20, os.cpu_count() or 4)) as pool:
        results = pool.map(run_job, jobs)

    print(f"\nAll done. {len(results)} traces saved to {OUT}/")
