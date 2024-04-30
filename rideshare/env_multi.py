import functools
import random
from copy import copy

import numpy as np
import torch
from pettingzoo.utils.env import ParallelEnv
from torch.utils.tensorboard import SummaryWriter
import wandb
writer = SummaryWriter()
class MultiRideshareEnv(ParallelEnv):
    metadata = {
        "name": "multirideshare_v0",
        "render_modes": ["text"]
    }

    def __init__(self, **kwargs):
        super(MultiRideshareEnv, self).__init__()
        self.render_mode = 'text'
        self.possible_agents = ["U", "L"]
        # by how much they may change their rates/commission per iteration
        self.change_rate = kwargs["change_rate"]
        self.max_rate = kwargs["max_rate"]  # Maximum rates
        self.OD = kwargs["OD"]  # the OD matrix
        self.C = kwargs["C"]  # Distance matrix
        self.N = self.OD.shape[0]  # number of nodes in OD matrix
        self.max_timestep = kwargs["max_timestep"]  # max simulation step for one episode
        self.lbd = kwargs["lbd"]  # lambda > 0
        self.rp = kwargs["rp"]  # public price > 0
        self.g = kwargs["g"]  # gas cost > 0
        self.num_D_samples = kwargs["num_D_samples"]  # number of samples to generate for drivers at each iter
        self.a_d = kwargs["a_d"]  # delta a, by how much drivers may deviate at each iteration
        self.p_d = kwargs["p_d"]  # delta p, by how much passengers may deviate at each iteration
        self.init_passenger_distribution = kwargs["init_passenger_distribution"]
        self.init_driver_distribution = kwargs["init_driver_distribution"]
        self.passenger_population = sum(self.init_passenger_distribution)
        self.driver_population = sum(self.init_driver_distribution)
        self.alpha = kwargs["alpha"]
        self.mode = kwargs["mode"]              # delta for delta actions and interp for interpolations
        self.validate()

    def validate(self):
        assert sum(self.init_driver_distribution) <= self.driver_population, "Drivers cannot exceed population size"
        assert sum(self.init_passenger_distribution) <= self.passenger_population, "Passengers cannot exceed population size"
        assert isinstance(self.OD, np.ndarray), "OD matrix must be an array"
        assert len(self.OD.shape) == 2, "OD matrix must be 2D"
        assert self.OD.shape[0] == self.OD.shape[1], "OD matrix is a square matrix"
        # assert np.sum(self.OD) == 1., "OD matrix must sum to 1"
        # assert np.all(np.diag(self.OD) == 0.), "Diagonal of OD matrix is 0"

    def reset(self, seed, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        # Rates randomly initialized between the price range for each node
        # RU = np.random.uniform(low=self.g, high=self.max_rate, size=(self.N, self.N))  # Rate for each edge
        # # Commissions randomly initialized but greater than gas cost and less than the rates
        # CU = np.random.uniform(low=self.g, high=RU, size=(self.N, self.N))
        # RL = np.random.uniform(low=self.g, high=self.max_rate, size=(self.N, self.N))
        # CL = np.random.uniform(low=self.g, high=RL, size=(self.N, self.N))

        # At each edge
        self.e = 0.1
        PU = np.clip(0., 1., np.random.normal(1/3., 0.01, (self.N, self.N)))
        np.fill_diagonal(PU, 0.)
        PL = np.clip(0., PU, np.random.normal(1/3., 0.01, (self.N, self.N)))
        np.fill_diagonal(PL, 0.)
        PP = 1 - PU - PL
        np.fill_diagonal(PL, 0.)
        # At each node, uniformly want to drive for each platform
        Au = np.clip(0, 1., np.random.normal(1 / 2., self.e, self.N))
        Al = 1 - Au
        init_state = (
            Au, Al, PU, PL, PP, copy(self.init_passenger_distribution),
            copy(self.init_driver_distribution)
        )
        self.state = init_state

        observations = {a: self.flatten_obs(self.state) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def towards(self, a, b, step):
        # move a towards b in step increment.
        if b > a:
            if b - a <= step:
                return b
            return a + step
        if a - b <= step:
            return b
        return a - step

    def allocate(self, RU, CU, RL, CL, Au, Al, PU, PL, PP, passenger_distribution, driver_distribution):
        Au_opt, Al_opt, PU_opt, PL_opt, PP_opt = Au, Al, PU, PL, PP
        max_profit = -9999
        rp = self.rp  # public transit rate
        g = self.g  # gas cost
        for _ in range(self.num_D_samples):
            if _ > 0:
                Au_d = np.random.uniform(-self.a_d, self.a_d, size=Au.shape)
            else:
                Au_d = 0.
            # induced rational response
            Au_ = np.clip(Au + Au_d, 0., 1.)
            Al_ = 1. - Au_
            profit = 0.
            PU_, PP_, PL_ = np.zeros_like(PU), np.zeros_like(PP), np.zeros_like(PL)
            # For each edge, calculate the new allocation
            for i in range(self.N):
                # Lambda actually depends upon the number of drivers at a node and number of passengers at the node.
                # The way it's used in the theoretical derivations consider these numbers to be the same
                # so it cancels out. But here, this may not be the case. To be consistent with the formulae,
                # we must appropriately account for the wait cost by scaling lambda for each node.
                num_passengers = passenger_distribution[i]
                num_drivers = driver_distribution[i]

                if num_passengers == 0.:
                    continue

                if num_drivers > 0:
                    lbd = self.lbd * num_passengers / num_drivers
                else:
                    lbd = -1

                # candidate allocations
                au_, al_ = Au_[i], Al_[i]
                for j in range(self.N):
                    if self.OD[i, j] == 0.: continue
                    if lbd == -1:
                        pu_ = 0.
                        pl_ = 0.
                        pp_ = 1.
                    elif lbd <= 0:
                        raise Exception("Lambda cannot be less or equal to 0 except -1")
                    else:
                        rl = RL[i, j]
                        ru = RU[i, j]
                        pu_ = (2 * lbd * au_ + al_ * au_ * (rl - ru) + au_ * (rp - ru)) / (2 * lbd * (au_ + al_ + 1))
                        pu_ = np.clip(pu_, 0., 1.)
                        pl_ = (2 * lbd * al_ + al_ * au_ * (ru - rl) + al_ * (rp - rl)) / (2 * lbd * (au_ + al_ + 1))
                        pl_ = np.clip(pl_, 0., 1. - pu_)
                        pp_ = 1 - pu_ - pl_
                    # Move the current allocation towards the optimal allocation for this candidate
                    pu_, pl_, pp_ = (
                        self.towards(PU[i, j], pu_, self.p_d),
                        self.towards(PL[i, j], pl_, self.p_d),
                        self.towards(PP[i, j], pp_, self.p_d))
                    PU_[i, j] = pu_
                    PL_[i, j] = pl_
                    PP_[i, j] = pp_
                    # We don't really care about the matching constraint
                    # calculate instantaneous profit for drivers
                    profit +=  self.C[i, j] * (pu_ * (CU[i, j] - g) + pl_ * (CL[i, j] - g))
                    # profit += num_passengers * self.C[i, j] * (pu_ * (CU[i, j] - g) + pl_ * (CL[i, j] - g))

            if profit > max_profit:
                max_profit = profit
                Au_opt, Al_opt, PU_opt, PL_opt, PP_opt = Au_, Al_, PU_, PL_, PP_
        # if max_profit < 0:
        #     raise Exception("Couldn't find a valid allocation")
        return Au_opt, Al_opt, PU_opt, PL_opt, PP_opt

    def adjust(self, RU, CU, RL, CL, Au, Al, PU, PL, PP, passenger_distribution, driver_distribution):
        # After the allocation has been calculated, we adjust the graph, and calculate the profits
        alpha = self.alpha
        passenger_wait_costs_U, passenger_wait_costs_L, passenger_wait_costs_P = np.zeros_like(RU), np.zeros_like(
            RU), np.zeros_like(RU)
        new_passenger_distribution, new_driver_distribution = np.copy(passenger_distribution), np.copy(
            driver_distribution)
        revenue_U, revenue_L = np.zeros_like(RU), np.zeros_like(RU)
        cost_U, cost_L = np.zeros_like(RU), np.zeros_like(RU)
        revenue_P = np.zeros_like(RU)

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                if passenger_distribution[i] == 0.:
                    continue
                e = (i, j)  # current edge
                # outgoing from this edge
                num_passengers_uber = passenger_distribution[i] * self.OD[e] * PU[e]
                num_passengers_lyft = passenger_distribution[i] * self.OD[e] * PL[e]
                num_passengers_public = passenger_distribution[i] * self.OD[e] * PP[e]
                num_drivers_uber = driver_distribution[i] * Au[i] * self.OD[e]
                num_drivers_lyft = driver_distribution[i] * Al[i] * self.OD[e]
                # matching constraint
                num_trips_uber = min(num_passengers_uber, num_drivers_uber)
                num_trips_lyft = min(num_passengers_lyft, num_drivers_lyft)
                num_trips_public = num_passengers_public  # Everyone can fit inside a public transit

                # calculate costs and profits for everyone
                revenue_U[e] += num_trips_uber * self.C[e] * RU[e]
                cost_U[e] += num_trips_uber * self.C[e] * CU[e]
                revenue_L[e] += num_trips_lyft * self.C[e] * RL[e]
                cost_L[e] += num_trips_lyft * self.C[e] * CL[e]
                revenue_P[e] += num_passengers_public * self.C[e] * self.rp

                if num_drivers_lyft > 0.:
                    passenger_wait_costs_L[e] = self.lbd * num_passengers_lyft / num_drivers_lyft
                else:
                    passenger_wait_costs_L[e] = 1e9     # high value

                if num_drivers_uber> 0.:
                    passenger_wait_costs_U[e] = self.lbd * num_passengers_uber / num_drivers_uber
                else:
                    passenger_wait_costs_U[e] = 1e9

                passenger_wait_costs_P[e] += self.lbd * num_passengers_public / self.passenger_population

                # distribute the graph
                passengers_moved = alpha * (num_trips_uber + num_trips_lyft + num_trips_public)
                new_passenger_distribution[j] += passengers_moved
                new_passenger_distribution[i] -= passengers_moved
                new_passenger_distribution[i] = max(new_passenger_distribution[i], 0.)

                drivers_moved = alpha * (num_trips_uber + num_trips_lyft)
                new_driver_distribution[j] += drivers_moved
                new_driver_distribution[i] -= drivers_moved
                new_driver_distribution[i] = max(new_driver_distribution[i], 0.)
                # add / remove new participants
                # TODO: figure out if we're adding / removing. If we are, we have to adjust the population parameter.


        # These are instanteneous profits
        profits_U = (revenue_U - cost_U)
        profits_L = (revenue_L - cost_L)
        update_dict = dict()
        if self.log:
            for i in range(self.N):
                update_dict.update({f'Au_{i}': Au[i], f'Al_{i}': Al[i]})
                update_dict.update({f'passenger_distribution{i}': new_passenger_distribution[i],
                           f'driver_distribution{i}': new_driver_distribution[i]})
                for j in range(self.N):
                    if i == j: continue
                    e = (i, j)
                    tag = f'_{i}_{j}'
                    update_dict.update({f'RU{tag}': RU[e], f'CU{tag}': CU[e]})
                    update_dict.update({f'RL{tag}': RL[e], f'CL{tag}': CL[e]})
                    update_dict.update({f'PU{tag}': PU[e], f'PL{tag}': PL[e], f'PP{tag}': PP[e]})
                    update_dict.update({f'profits_U{tag}': profits_U[e], f'profits_L{tag}': profits_L[e]})
            update_dict.update({'profits_U': np.sum(profits_U), 'profits_L': np.sum(profits_L)})
        if self.log:
            wandb.log(update_dict)
        return np.sum(profits_U), np.sum(profits_L), new_passenger_distribution, new_driver_distribution

    def step(self, actions, log):
        self.log = log
        self.timestep += 1
        # deltas for the agents
        RU_d = actions["U"][:self.N, :]
        CU_d = actions["U"][self.N:, :]
        RL_d = actions["L"][:self.N, :]
        CL_d = actions["L"][self.N:, :]
        Au, Al, PU, PL, PP, passenger_distribution, driver_distribution = self.state
        if self.mode == "interp":
            interp = lambda r: r * (self.max_rate - self.g) + self.g
            RU = interp(RU_d)
            CU = interp(CU_d)
            RL = interp(RL_d)
            CL = interp(CL_d)
        elif self.mode == "delta":
            raise Exception("Unknown mode")

        RU = np.clip(RU, a_min=self.g, a_max=self.max_rate)
        CU = np.clip(CU, a_min=self.g, a_max=self.max_rate)
        RL = np.clip(RL, a_min=self.g, a_max=self.max_rate)
        CL = np.clip(CL, a_min=self.g, a_max=self.max_rate)

        # Now the drivers and passengers decide on the incremental allocation
        Au, Al, PU, PL, PP = self.allocate(RU, CU, RL, CL, Au, Al, PU, PL, PP, passenger_distribution,
                                           driver_distribution)
        profit_U, profit_L, passenger_distribution, driver_distribution = self.adjust(RU, CU, RL, CL, Au, Al, PU, PL,
                                                                                      PP, passenger_distribution,
                                                                                      driver_distribution)
        self.state = (Au, Al, PU, PL, PP, passenger_distribution, driver_distribution)
        observations = {a: self.flatten_obs(self.state) for a in self.agents}

        # Scale reward for stability
        rewards = {"U": np.sum(profit_U)    * self.alpha,
                   "L": np.sum(profit_L)    * self.alpha}

        # agents never really terminate.
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, rewards, terminations, truncations, infos

    def flatten_obs(self, obs):
        return np.concatenate([o.flatten() for o in obs])

    def render(self):
        pass

    @functools.lru_cache()
    def observation_space(self, agent):
        # each agent can observe the following:
        #   the prevailing rate and commission of both agents (ru, cu, rl, cl)
        #   current supply / demand market activity (au, al, pu, pl, pp)
        # RU, CU, RL, CL, Au, Al, PU, PL, PP, passenger_distribution, driver_distribution
        from gymnasium.spaces import Box
        l = self.g
        h = self.max_rate
        # The true space looks like this but using a dict space is too much trouble when it comes to training
        # via not only pettingzoo but also stablebaselines. So we stick with trivial spaces like Box
        # Dict({
        #     "RU": Box(l, h, shape2d),
        #     "CU": Box(l, h, shape2d),
        #     "RL": Box(l, h, shape2d),
        #     "CL": Box(l, h, shape2d),
        #     "Au": Box(0., 1., shape1d),
        #     "Al": Box(0., 1., shape1d),
        #     "PU": Box(0., 1., shape2d),
        #     "PL": Box(0., 1., shape2d),
        #     "PP": Box(0., 1., shape2d),
        #     "passenger_distribution": Box(0., self.passenger_population, shape1d),
        #     "driver_distribution": Box(0., self.driver_population, shape1d)
        # })
        return Box(low=np.array( [0.] * (3 * self.N ** 2 + 4 * self.N)),
                   high=np.array( [1.] * (3 * self.N ** 2 + 2 * self.N) +
                        [self.passenger_population] * self.N + [self.driver_population] * self.N ))

    @functools.lru_cache()
    def action_space(self, agent):
        from gymnasium.spaces import Box
        # for each agent, the action is the change in rate and commission per iteration
        shape = (2*self.N, self.N)
        return Box(0, 1, shape=shape)
