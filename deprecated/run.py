from __future__ import annotations

import glob
import os
import time
import wandb
import numpy as np
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
import supersuit as ss
from rideshare.env_multi import MultiRideshareEnv

run = None

def train(env_fn, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    env = env_fn(**env_kwargs)
    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    # Use a CNN policy if the observation space is visual
    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        batch_size=256,
        gamma=env_kwargs["ppo_args"]["gamma"]
        # tensorboard_log=f"runs/{run.id}"
    )

    model.learn(total_timesteps=env_kwargs["ppo_args"]["steps"], callback=WandbCallback())

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)


    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward


if __name__ == "__main__":
    env_fn = MultiRideshareEnv

    OD = [[0., 0.3, 0.7],
          [0.4, 0., 0.6],
          [0.1, 0.9, 0.]]
    C = [[0., 4., 5.],
         [4., 0., 2.,],
         [5., 2., 0.]]
    init_passenger_distribution = [2000., 3000., 5000.]
    init_driver_distribution = [1000., 5000., 2000.]
    # Set vector_state to false in order to use visual observations (significantly longer training time)
    ts = 2*864000
    env_kwargs = {
        "OD":                                       np.array(OD),
        "C":                                        np.array(C),
        "init_passenger_distribution":              np.array(init_passenger_distribution),
        "init_driver_distribution":                 np.array(init_driver_distribution),
        "change_range":     1,                      # only changes 1 per second
        "max_rate":         30.,                    # maximum charged per mile is 30
        "max_timestep":     ts ,                    # seconds in a day
        "lbd":              3.,                     # lambda
        "rp":               12.,                    # maximum rate public transportation charges
        "g":                5.,                     # gas cost
        "num_D_samples":    10,                     # random samples generated by drivers
        "a_d":              1e-1,                   # at each second, can only change very little.
        "p_d":              1e-1,                   # at each second, passengers can deviate by.
        "alpha":            1e-4,                   # step parameter for continuous term
        "ppo_args":         {
                                "steps": ts,
                                "gamma": 0.99,
                            }
    }
    run = wandb.init(
        # Set the project where this run will be logged
        project="Rideshare Multi",
        # Track hyperparameters and run metadata
        config=env_kwargs,
        # sync_tensorboard=True,
        mode="disabled"
    )

    # Train a model (takes ~5 minutes on a laptop CPU)
    train(env_fn, seed=0, **env_kwargs)

    # # Evaluate 10 games (takes ~10 seconds on a laptop CPU)
    # eval(env_fn, num_games=10, render_mode=None, **env_kwargs)
