import math
from copy import deepcopy
import torch
import torch.nn.functional as F
import numpy as np
from base_agent import BaseAgent


class DQNAgent(BaseAgent):
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
        self.config = config

    def sync_weight(self):
        """Synchronize the weight for the target network."""
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def set_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay_steps)

    def get_action(self, state):
        self.epsilon = self.set_epsilon()
        self.steps_done += 1
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            qvals, _ = self.q_net(state)
        action = qvals.max(1)[1].item()

        if torch.rand(1).item() < self.epsilon:
            return np.random.randint(self.config['action_shape'])

        return action

    def get_action_eval(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            qvals, _ = self.q_net(state)
        action = qvals.max(1)[1].item()
        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        if self.double_q:
            self.q_net.eval()
            with torch.no_grad():
                next_actions = self.q_net(next_states)[0].max(dim=1)[1]
                next_q, _ = self.target_q_net(next_states)
            max_next_q = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            expected_q = (rewards.squeeze(1) + self.gamma * max_next_q) * (1 - dones)

            self.q_net.train()
            curr_q = self.q_net(states)[0].gather(1, actions.unsqueeze(1))
            curr_q = curr_q.squeeze(1)
        else:
            with torch.no_grad():
                next_q, _ = self.target_q_net(next_states)
            max_next_q = next_q.max(dim=1)[0].detach()
            expected_q = (rewards.squeeze(1) + self.gamma * max_next_q) * (1 - dones)

            self.q_net.train()
            curr_q = self.q_net(states)[0].gather(1, actions.unsqueeze(1))
            curr_q = curr_q.squeeze(1)


        # Compute Huber loss to handle outliers rubostly
        # loss = F.smooth_l1_loss(curr_q, expected_q)
        # MSE loss
        loss = F.mse_loss(curr_q, expected_q)
        return loss

    def learn(self, batch, **kwargs):
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self._target_update_freq == 0:
            self.sync_weight()

        return {'q_net_loss': loss.item(), 'eps': self.epsilon}
