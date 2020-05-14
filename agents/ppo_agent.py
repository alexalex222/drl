#######################################################################
# Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and             #
# Klimov, O., 2017. Proximal policy optimization algorithms.          #
# arXiv preprint arXiv:1707.06347.                                    #
#######################################################################

from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from agents.base_agent import BaseAgent


class ProximalPolicyOptimizationAgent(BaseAgent):
    def __init__(self,
                 actor,
                 critic,
                 optimizer,
                 dist_fn,
                 config,
                 **kwargs):
        super().__init__(**kwargs)
        self.actor = actor
        self.critic = critic
        self._w_vf = config['value_loss_weight']
        self._w_ent = config['entropy_loss_weight']
        self.gamma = config['discount_factor']
        self.steps_done = 0
        self.optimizer = optimizer
        self.dist_fn = dist_fn
        # device: cpu or gpu
        self.device = torch.device(config['device'])
        # gradient clip
        self._grad_clip = config['grad_clip']
        self.config = config

    def _calculate_returns(self, reward_trajectory, done_trajectory):
        returns = deepcopy(reward_trajectory)
        last = 0
        for i in range(len(returns) - 1, -1, -1):
            if not done_trajectory[i]:
                returns[i] += self.gamma * last
            last = returns[i]

        return returns

    def get_action(self, state):
        self.steps_done += 1
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.dist_fn == torch.distributions.Categorical:
                logits = self.actor(state)
                dist = self.dist_fn(logits=logits)
            elif self.dist_fn == torch.distributions.Normal:
                mean, std = self.actor(state)
                dist = self.dist_fn(mean, std)
            else:
                raise ValueError('Wrong distribution function!')
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        action = action.detach().cpu().numpy()[0]
        return action

    def get_action_eval(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.dist_fn == torch.distributions.Categorical:
                logits = self.actor(state)
                dist = self.dist_fn(logits=logits)
            elif self.dist_fn == torch.distributions.Normal:
                mean, std = self.actor(state)
                dist = self.dist_fn(mean, std)
            else:
                raise ValueError('Wrong distribution function!')
            action = dist.sample()
        action = action.detach().cpu().numpy()[0]
        return action

    def learn(self, full_batch, **kwargs):
        states_traj, actions_traj, rewards_traj, next_states_traj, dones_traj = full_batch
        returns_trajectory = self._calculate_returns(rewards_traj, dones_traj)
        # returns : shape [batch_size x 1]
        returns_trajectory = torch.tensor(returns_trajectory, dtype=torch.float32).to(self.device)
        returns_trajectory = returns_trajectory.unsqueeze(-1)
        # states: shape [batch_size x state_dim]
        states_traj = torch.tensor(states_traj, dtype=torch.float32).to(self.device)
        # states: shape [batch_size]
        actions_traj = torch.tensor(actions_traj, dtype=torch.float32).to(self.device)

        # compute old log_prob and old value
        with torch.no_grad():
            old_value_traj = self.critic(states_traj).detach()
            advantages_traj = returns_trajectory - old_value_traj
            if self.dist_fn == torch.distributions.Categorical:
                logits = self.actor(states_traj)
                dist = self.dist_fn(logits=logits)
                old_log_prob_traj = dist.log_prob(actions_traj).detach()
            elif self.dist_fn == torch.distributions.Normal:
                mean, std = self.actor(states_traj)
                dist = self.dist_fn(mean, std)
                old_log_prob_traj = dist.log_prob(actions_traj).sum(-1).detach()
            else:
                raise ValueError('Wrong distribution function!')


        train_dataset = TensorDataset(states_traj,
                                      actions_traj,
                                      returns_trajectory,
                                      old_log_prob_traj,
                                      advantages_traj)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        repeat = self.config['optimization_epochs']
        for _ in range(repeat):
            for idx, (states, actions, returns, old_log_prob, advantages) in enumerate(train_loader):
                if self.dist_fn == torch.distributions.Categorical:
                    logits = self.actor(states)
                    dist = self.dist_fn(logits=logits)
                    log_prob = dist.log_prob(actions)
                elif self.dist_fn == torch.distributions.Normal:
                    mean, std = self.actor(states)
                    dist = self.dist_fn(mean, std)
                    log_prob = dist.log_prob(actions).sum(-1)
                else:
                    raise ValueError('Wrong distribution function!')


                ratio = (log_prob - old_log_prob).exp()
                obj = ratio * advantages
                obj_clipped = ratio.clamp(1.0 - self.config['ratio_clip'],
                                          1.0 + self.config['ratio_clip']) * advantages
                policy_loss = -torch.min(obj, obj_clipped).mean()
                entropy_loss = dist.entropy().mean()
                values = self.critic(states)
                value_loss = F.mse_loss(returns, values)

                total_loss = policy_loss + self._w_vf * value_loss - self._w_ent * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) +
                                         list(self.critic.parameters()),
                                         self._grad_clip)
                self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
        }
