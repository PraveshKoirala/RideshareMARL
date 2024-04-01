import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from pettingzoo import ParallelEnv


class RideshareEnv(ParallelEnv):
    metadata = {
        "name": "rideshare_v0",
    }

    def __init__(self, **kwargs):
        super(RideshareEnv, self).__init__()
        self.possible_agents = ["U", "L"]
        # by how much they may change their rates/commission per iteration
        self.change_range = kwargs["change_range"]
        # total bounds on their rates/commission (theoretically unbounded)
        self.price_range = kwargs["price_range"]
        # number of nodes?
        self.N = kwargs["N"]
        self.max_timestep = kwargs["max_timestep"]
        self.lbd = kwargs["lbd"]
        self.rp = kwargs["rp"]
        self.g = kwargs["g"]
        self.num_D_samples = kwargs["num_D_samples"]
        self.a_d = kwargs["a_d"]
        self.p_d = kwargs["p_d"]

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        while True:
            ru = np.random.uniform(low=self.price_range[0], high=self.price_range[1])
            cu = np.random.uniform(low=self.price_range[0], high=self.price_range[1])
            rl = np.random.uniform(low=self.price_range[0], high=self.price_range[1])
            cl = np.random.uniform(low=self.price_range[0], high=self.price_range[1])
            pu = np.random.uniform(low=0., high=1.)
            pl = np.random.uniform(low=0., high=1-pu)
            au = np.random.uniform(low=0., high=pu+pl)
            al = np.random.uniform(low=0., high=pu + pl - au)
            pp = np.random.uniform(low=0., high=1.-pu-pl)
            init_state =(
                    ru, cu, rl, cl, au, al, pu, pl, pp
            )
            self.state = init_state
            try:
                self.allocate(ru, cu, rl, cl, pu, pl, au, al, pp, True)
                break
            except:
                continue

        observations = {a: init_state for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def towards(self, a, b, step):
        # move a towards b in step increment.
        if b>a:
            if b-a <= step:
                return b
            return a+step
        if a-b <= step:
            return b
        return a-step

    def allocate(self, ru, cu, rl, cl, au, al, pu, pl, pp, init=False):
        au_opt, al_opt, pu_opt, pl_opt, pp_opt = None, None, None, None, None
        max_profit = -9999
        l = self.lbd
        rp = self.rp
        g = self.g
        for _ in range(self.num_D_samples):
            if _ > 0 and init == False:
                au_d, al_d = np.random.uniform(-self.a_d, self.a_d, size=(2,))
            else:
                au_d , al_d = 0., 0.
            # induced rational response
            au_ = max(min(au + au_d, 1.), 0)
            al_ = al + al_d


            pu_ = (2*l*au_ + al_ * au_ * (rl-ru) + au_ * (rp - ru)) / (2*l*(au_+al_+1))
            pu_ = max(min(pu_, 1.), 0.)
            pl_ = (2*l*al_ + al_ * au_ * (ru-rl) + al_ * (rp - rl)) / (2*l*(au_+al_+1))
            pl_ = max(min(pl_, 1.-pu_), 0.)
            pp_ = 1-pu_- pl_
            au_, al_, pu_, pl_, pp_ = (
                self.towards(au, au_, self.a_d),
                self.towards(al, al_, self.a_d),
                self.towards(pu, pu_, self.p_d),
                self.towards(pl, pl_, self.p_d),
                self.towards(pp, pp_, self.p_d))
            if au_ + al_ > pu_ + pl_: continue # Matching constraint is violated.
            # calculate profits
            profit = pu_ * (cu - g) + pl_ * (cl - g)
            if profit > max_profit:
                max_profit = profit
                au_opt, al_opt, pu_opt, pl_opt, pp_opt = au_, al_, pu_, pl_, pp_
        if max_profit < 0:
            raise Exception("Couldn't find a valid allocation")
        return au_opt, al_opt, pu_opt, pl_opt, pp_opt

    def step(self, actions):
        self.timestep += 1
        # deltas for the agents
        ru_d, cu_d = actions["U"]
        rl_d, cl_d = actions["L"]
        ru, cu, rl, cl, au, al, pu, pl, pp = self.state

        ru += ru_d
        cu += cu_d
        rl += rl_d
        cl += cl_d

        # Now the drivers and passengers decide on the incremental allocation
        au, al, pu, pl, pp = self.allocate(ru, cu, rl, cl, au, al, pu, pl, pp)

        self.state = (ru, cu, rl, cl, au, al, pu, pl, pp)
        observations = {a: self.state for a in self.agents}
        rewards = {"U": pu * (ru - cu), "L": pl * (rl - cl) }
        # agents never really terminate.
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    @functools.lru_cache()
    def observation_space(self, agent):
        # each agent can observe the following:
        #   the prevailing rate and commission of both agents (ru, cu, rl, cl)
        #   current supply / demand market activity (au, al, pu, pl, pp)
        price_low, price_high = self.price_range
        return Box(low=np.array([price_low, price_low, price_low, price_low,
                                 0., 0., 0., 0., 0.]),
                   high=np.array([price_high, price_high, price_high, price_high,
                                 1., 1., 1., 1., 1.]),
                   dtype=np.float32)

    @functools.lru_cache()
    def action_space(self, agent):
        # for each agent, the action is the change in rate and commission per iteration
        return Box(*self.change_range, shape=(2,))