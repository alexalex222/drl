import math
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import gpytorch
import numpy as np
from agents.base_agent import BaseAgent
from networks.network_models import StandardGPModel


class GPAgent(BaseAgent):
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
        # batch index
        self.batch_index = torch.arange(config['batch_size']).long().to(config['device'])
        # inducing batch index
        self.inducing_batch_index = torch.arange(config['inducing_size']).long().to(config['device'])
        # gradient update frequency
        self._grad_update_freq = config['grad_update_freq']
        # gradient clip
        self._grad_clip = config['grad_clip']
        # q net loss
        self.q_net_loss = 0
        # likelihood function
        self.likelihoods = [gpytorch.likelihoods.GaussianLikelihood().to(self.device)
                            for _ in range(config['action_shape'])]
        for i in range(config['action_shape']):
            self.likelihoods[i].initialize(noise=1e-4)
        # gp layer
        self.gp_layers = [StandardGPModel(train_x=None,
                                          train_y=None,
                                          likelihood=self.likelihoods[i],
                                          kernel_type='linear').to(self.device)
                          for i in range(config['action_shape'])]
        for i in range(config['action_shape']):
            self.likelihoods[i].eval()
            self.gp_layers[i].eval()
        # count of greedy action from posterior sampling
        self.greedy_selection = 0
        self.config = config

    def sync_weight(self):
        """Synchronize the weight for the target network."""
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def set_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay_steps)

    def get_action(self, state):
        self.steps_done += 1
        for i in range(self.config['action_shape']):
            if self.gp_layers[i].train_inputs is None or self.gp_layers[i].train_targets is None:
                return np.random.randint(self.config['action_shape'])

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            q, h = self.q_net(state)
        q_posterior = np.empty([self.config['action_shape']])
        for i in range(self.config['action_shape']):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                q_posterior_sample = self.gp_layers[i](h.clone().detach()).sample().item()
            q_posterior[i] = q_posterior_sample
        action = np.argmax(q_posterior)
        action_greedy = torch.argmax(q, dim=1).item()
        if action == action_greedy:
            self.greedy_selection = self.greedy_selection + 1
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
        # states: shape [batch_size x state_dim]
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        # states: shape [batch_size]
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        # rewards: shape [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # next_states: shape [batch_size x state_dim]
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        # dones: shape [states]
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # compute the q values for next states
        with torch.no_grad():
            next_q, _ = self.target_q_net(next_states)

        if self.double_q:
            self.q_net.eval()
            with torch.no_grad():
                # best_actions: shape [batch_size]
                best_actions = torch.argmax(self.q_net(next_states)[0], dim=1)
            # max_next_q: shape [batch_size]
            max_next_q = next_q[self.batch_index, best_actions]
            # expected_q: shape [batch_size]
            expected_q = (rewards + self.gamma * max_next_q) * (torch.tensor([1]).to(self.device) - dones)

            self.q_net.train()
            # curr_q_all: shape [batch_size x action_dim]
            curr_q_all, _ = self.q_net(states)
            # curr_q: shape[batch_size]
            curr_q = curr_q_all[self.batch_index, actions]
        else:
            # max_next_q: shape [batch_size]
            max_next_q = next_q.max(dim=1)[0].detach()
            # expected_q: [batch_size]
            expected_q = (rewards + self.gamma * max_next_q) * (torch.tensor([1]).to(self.device) - dones)

            self.q_net.train()
            # curr_q: shape [batch_size]
            curr_q = self.q_net(states)[0][self.batch_index, actions]

        # Compute Huber loss to handle outliers robustly
        loss = F.smooth_l1_loss(curr_q, expected_q)
        # MSE loss
        # loss = F.mse_loss(curr_q, expected_q)
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

    def update_gp(self, inducing_batch):
        if self.steps_done % self._grad_update_freq == 0:
            states, actions, rewards, next_states, dones = inducing_batch
            # states: shape [batch_size x state_dim]
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            # states: shape [batch_size]
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            # rewards: shape [batch_size]
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            # next_states: shape [batch_size x state_dim]
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            # dones: shape [states]
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

            self.q_net.eval()
            # compute the q values for next states
            with torch.no_grad():
                next_q, _ = self.target_q_net(next_states)
                next_q_temp, features = self.q_net(next_states)

            if self.double_q:
                # best_actions: shape [batch_size]
                best_actions = torch.argmax(next_q_temp, dim=1)
            else:
                # max_next_q: shape [batch_size]
                best_actions = torch.argmax(next_q, dim=1)
            next_q[self.inducing_batch_index, best_actions] = \
                (rewards + self.gamma * next_q[self.inducing_batch_index, best_actions]) * \
                (torch.tensor([1]).to(self.device) - dones)

            for i in range(self.config['action_shape']):
                x_train = features.clone().detach()
                y_train = next_q[:, i].clone().detach()
                self.gp_layers[i].set_train_data(inputs=x_train, targets=y_train, strict=False)


