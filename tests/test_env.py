from rideshare.env import RideshareEnv

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = RideshareEnv(**{
        "change_range": (-1, 1),
        "price_range": (0, 20),
        "N": 1,
        "max_timestep": 1000,
        "lbd": 3,
        "rp": 4,
        "g": 1,
        "num_D_samples": 10,
        "a_d" : 0.05,
        "p_d" : 0.05
    })
    parallel_api_test(env, num_cycles=1_000_000)