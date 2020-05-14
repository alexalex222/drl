#######################################################################
# Bellemare, M.G., Dabney, W. and Munos, R., 2017, August.            #
# A distributional perspective on reinforcement learning.             #
# In Proceedings of ICML-2017                                         #
#######################################################################


from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseAgent


class CategoricalDQNAgent(BaseAgent):
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
        # atoms
        self.atoms = torch.linspace(config['v_min'], config['v_max'], steps=config['num_atoms']).to(config['device'])
        # delta between atoms
        self.delta_atom = (config['v_max'] - config['v_min']) / float(config['num_atoms'])
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
            q_prob, _ = self.q_net(state)
        q = (q_prob * self.atoms).sum(-1)
        action = q.max(dim=1)[1].item()
        return action

    def get_action_eval(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            q_prob, _ = self.q_net(state)
        q = (q_prob * self.atoms).sum(-1)
        action = q.max(1)[1].item()
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
            next_q_prob, _ = self.target_q_net(next_states)
        next_q_prob = next_q_prob.detach()
        next_q = (next_q_prob * self.atoms).sum(-1)
        next_a = torch.argmax(next_q, dim=-1)
        next_q_prob = next_q_prob[self.batch_index, next_a, :]

        atoms_next = rewards + self.gamma * (1 - dones) * self.atoms.view(1, -1)
        atoms_next.clamp_(self.config['v_min'], self.config['v_max'])
        b = (atoms_next - self.config['v_min']) / self.delta_atom
        l = b.floor()
        u = b.ceil()
        d_m_l = (u + (l == u).float() - b) * next_q_prob
        d_m_u = (b - l) * next_q_prob
        target_prob = torch.zeros(next_q_prob.shape).to(self.device)

        for i in range(target_prob.size(0)):
            target_prob[i].index_add_(0, l[i].long(), d_m_l[i])
            target_prob[i].index_add_(0, u[i].long(), d_m_u[i])

        self.q_net.train()
        _, log_prob = self.q_net(states)
        log_prob = log_prob[self.batch_index, actions, :]
        loss = -(target_prob * log_prob).sum(-1).mean()
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
