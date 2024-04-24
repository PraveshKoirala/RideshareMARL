# Based entirely on https://github.com/seolhokim/Mujoco-Pytorch/

from abc import *

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import ReplayBuffer, make_mini_batch, convert_to_tensor


class NetworkBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super(NetworkBase, self).__init__()

    @abstractmethod
    def forward(self, x):
        return x


class Network(NetworkBase):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function=torch.relu,
                 last_activation=None):
        super(Network, self).__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)
        layers = ([nn.Linear(layers_unit[idx], layers_unit[idx + 1]) for idx in range(len(layers_unit) - 1)])
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(layers_unit[-1], output_dim)
        self.network_init()

    def forward(self, x):
        return self._forward(x)

    def _forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.last_layer(x)
        if self.last_activation != None:
            x = self.last_activation(x)
        return x

    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()


class Actor(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function=torch.tanh,
                 last_activation=None, trainable_std=False):
        super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation)
        self.trainable_std = trainable_std
        if self.trainable_std == True:
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        mu = self._forward(x)
        if self.trainable_std == True:
            std = torch.exp(self.logstd)
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu, std


class Critic(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation=None):
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation)

    def forward(self, *x):
        x = torch.cat(x, -1)
        return self._forward(x)


class PPO(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(PPO, self).__init__()
        self.args = args

        self.data = ReplayBuffer(action_prob_exist=True, max_size=self.args.traj_length, state_dim=state_dim,
                                 num_action=action_dim)
        self.actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, \
                           self.args.activation_function, self.args.last_activation_actor, self.args.trainable_std)
        self.critic = Critic(self.args.layer_num, state_dim, 1, \
                             self.args.hidden_dim, self.args.activation_function, self.args.last_activation_critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)

        self.writer = writer
        self.device = device

    def get_action(self, x):
        mu, sigma = self.actor(x)
        return mu, sigma

    def v(self, x):
        return self.critic(x)

    def put_data(self, transition):
        self.data.put_data(transition)

    # def empty_data(self):
    #     self.data.

    def get_gae(self, states, rewards, next_states, dones):
        values = self.v(states).detach()
        td_target = rewards + self.args.gamma * self.v(next_states) * (1 - dones)
        delta = td_target - values
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0
        for idx in reversed(range(len(delta))):
            if dones[idx] == 1:
                advantage = 0.0
            advantage = self.args.gamma * self.args.lambda_ * advantage + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantages = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
        return values, advantages

    def train_net(self, n_epi):
        data = self.data.sample(shuffle=False)
        states, actions, rewards, next_states, dones, old_log_probs = convert_to_tensor(self.device, data['state'],
                                                                                        data['action'], data['reward'],
                                                                                        data['next_state'],
                                                                                        data['done'], data['log_prob'])

        old_values, advantages = self.get_gae(states, rewards, next_states, dones)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-3)

        for i in range(self.args.train_epoch):
            for state, action, old_log_prob, advantage, return_, old_value \
                    in make_mini_batch(self.args.batch_size, states, actions, \
                                       old_log_probs, advantages, returns, old_values):
                curr_mu, curr_sigma = self.get_action(state)
                value = self.v(state).float()
                curr_dist = torch.distributions.Normal(curr_mu, curr_sigma)
                entropy = curr_dist.entropy() * self.args.entropy_coef
                curr_log_prob = curr_dist.log_prob(action).sum(1, keepdim=True)

                # policy clipping
                ratio = torch.exp(curr_log_prob - old_log_prob.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.args.max_clip, 1 + self.args.max_clip) * advantage
                actor_loss = (-torch.min(surr1, surr2) - entropy).mean()

                # value clipping (PPO2 technic)
                old_value_clipped = old_value + (value - old_value).clamp(-self.args.max_clip, self.args.max_clip)
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
                critic_loss = 0.5 * self.args.critic_coef * torch.max(value_loss, value_loss_clipped).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                self.critic_optimizer.step()

                if self.writer != None:
                    self.writer.add_scalar("loss/actor_loss", actor_loss.item(), n_epi)
                    self.writer.add_scalar("loss/critic_loss", critic_loss.item(), n_epi)
