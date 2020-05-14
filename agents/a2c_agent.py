from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from agents.base_agent import BaseAgent


class AdvantageActorCriticAgent(BaseAgent):
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
            logits = self.actor(state)
            dist = self.dist_fn(logits=logits)
            action = dist.sample()
        action = action.detach().cpu().numpy()[0]
        return action

    def get_action_eval(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(state)
            dist = self.dist_fn(logits=logits)
            action = dist.sample()
        action = action.detach().cpu().numpy()[0]
        return action

    def learn(self, full_batch, repeat=1, **kwargs):
        states_traj, actions_traj, rewards_traj, next_states_traj, dones_traj = full_batch
        returns_trajectory = self._calculate_returns(rewards_traj, dones_traj)
        # states: shape [batch_size x state_dim]
        states_traj = torch.tensor(states_traj, dtype=torch.float32)
        # states: shape [batch_size]
        actions_traj = torch.tensor(actions_traj, dtype=torch.float32)
        # rewards: shape [batch_size]
        rewards_traj = torch.tensor(rewards_traj, dtype=torch.float32)
        # next_states: shape [batch_size x state_dim]
        next_states_traj = torch.tensor(next_states_traj, dtype=torch.float32)
        # dones: shape [batch_size]
        dones_traj = torch.tensor(dones_traj, dtype=torch.float32)
        # returns : shape [batch_size]
        returns_trajectory = torch.tensor(returns_trajectory, dtype=torch.float32)
        train_dataset = TensorDataset(states_traj,
                                      actions_traj,
                                      rewards_traj,
                                      next_states_traj,
                                      dones_traj,
                                      returns_trajectory)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        for _ in range(repeat):
            for idx, (states, actions, rewards, next_states, dones, returns) in enumerate(train_loader):
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)
                returns = returns.to(self.device)

                self.optimizer.zero_grad()
                # [batch_size x state_shape]
                logits = self.actor(states)
                dist = self.dist_fn(logits=logits)
                # [batch_size x 1]
                values = self.critic(states)
                a_loss = -(dist.log_prob(actions) * (returns - values).detach()).mean()
                vf_loss = F.mse_loss(returns, values.reshape(-1))
                ent_loss = dist.entropy().mean()
                loss = a_loss + self._w_vf * vf_loss - self._w_ent * ent_loss
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) +
                                         list(self.critic.parameters()),
                                         self._grad_clip)
                self.optimizer.step()

        return {
            'total_loss': loss.item(),
            'policy_loss': a_loss.item(),
            'value_loss': vf_loss.item(),
            'entropy_loss': ent_loss.item(),
        }
