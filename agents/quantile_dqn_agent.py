#######################################################################
# Dabney, W., Rowland, M., Bellemare, M.G. and Munos, R., 2018, April.#
# Distributional reinforcement learning with quantile regression.     #
# In Thirty-Second AAAI Conference on Artificial Intelligence.        #
#######################################################################

from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseAgent


class QuantileDQNAgent(BaseAgent):
    def __init__(self,
                 q_net,
                 optimizer,
                 config,
                 **kwargs):
        super().__init__(**kwargs)
        # q network
        self.q_net = q_net
        # target q network
        self.target_q_net = deepcopy(self.q_net)
        self.target_q_net.eval()
        # discount factor
        self.gamma = config['discount_factor']
        # quantile weight
        self.quantile_weight = 1.0 / config['num_quantiles']
        # cumulative density
        self.cumulative_density = torch.tensor(
            (2 * np.arange(config['num_quantiles']) + 1) / (2.0 * config['num_quantiles']),
            dtype=torch.float32, device=config['device']).view(1, -1)
        # frequency to udpate target q net
        self._target_update_freq = config['target_update_freq']
        # epsilon start value in training
        self.eps_start = config['eps_start']
        # epsilon end value in training
        self.eps_end = config['eps_end']
        # epsilon decay rate
        self.eps_decay_steps = config['eps_decay_steps']
        # current epsilon value
        self.epsilon = config['eps_start']
        # optimizer for q_net
        self.optimizer = optimizer
        # device: cpu or gpu
        self.device = torch.device(config['device'])
        # number of completed steps
        self.steps_done = 0
        # flag to use double q learning
        self.double_q = config['double_q']
        # batch index
        self.batch_index = torch.arange(config['batch_size']).long().to(config['device'])
        # gradient update frequency
        self._grad_update_freq = config['grad_update_freq']
        # gradient clip
        self._grad_clip = config['grad_clip']
        # q net loss
        self.q_net_loss = 0
        self.config = config

    def sync_weight(self):
        """Synchronize the weight for the target network."""
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def set_epsilon(self):
        # eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay_steps)
        eps = max(self.eps_start + (self.eps_end - self.eps_start) * (self.steps_done / self.eps_decay_steps),
                  self.eps_end)
        return eps

    def get_action(self, state):
        self.steps_done += 1

        self.epsilon = self.set_epsilon()
        if torch.rand(1).item() < self.epsilon:
            return np.random.randint(self.config['action_shape'])

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(state).mean(-1)
        action = q.max(dim=1)[1].item()
        return action

    def get_action_eval(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            qvals, _ = self.q_net(state)
        action = qvals.max(1)[1].item()
        return action

    def huber(self, x, k=1.0):
        return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        # states: shape [batch_size x state_dim]
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        # states: shape [batch_size]
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        # rewards: shape [batch_size x 1]
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        # next_states: shape [batch_size x state_dim]
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        # dones: shape [batch_size x 1]
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

        # compute the q values for next states
        with torch.no_grad():
            next_q_quantiles = self.target_q_net(next_states).detach()
        next_actions = torch.argmax(next_q_quantiles.mean(-1), dim=-1)
        next_q_quantiles = next_q_quantiles[self.batch_index, next_actions, :]
        next_q_quantiles = rewards + self.config.discount * (1 - dones) * next_q_quantiles

        self.q_net.train()
        q_quantiles = self.q_net(states)
        q_quantiles = q_quantiles[self.batch_index, actions, :]
        next_q_quantiles = next_q_quantiles.t().unsqueeze(-1)
        diff = next_q_quantiles - q_quantiles
        loss = self.huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
        return loss

    def learn(self, batch, **kwargs):
        is_updated = False
        if self.steps_done % self._grad_update_freq == 0:
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self._grad_clip)
            self.optimizer.step()
            self.q_net_loss = loss.item()
            is_updated = True

        if self.steps_done % self._target_update_freq == 0:
            self.sync_weight()

        if is_updated:
            return {'q_net_loss': self.q_net_loss, 'eps': self.epsilon}
        else:
            return None
