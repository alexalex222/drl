import torch
from torch import nn
import torch.nn.functional as F
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

    def _calculate_returns(self, rollout):
        reward_trajectory, done_trajectory = rollout
        returns = reward_trajectory
        last = 0
        for i in range(len(returns) - 1, -1, -1):
            if not done_trajectory[i]:
                returns[i] += self.gamma * last
            last = returns[i]

        return returns

    def get_action(self, state):
        self.steps_done += 1
        with torch.no_grad():
            logits = self.actor(state)
            dist = self.dist_fn(logits=logits)
            action = dist.sample()
        return action

    def get_action_eval(self, state):
        with torch.no_grad():
            logits = self.actor(state)
            dist = self.dist_fn(logits=logits)
            action = dist.sample()
        return action

    def learn(self, full_batch, batch_size=None, repeat=1, **kwargs):
        for _ in range(repeat):
            for b in full_batch.sample(batch_size):
                states, actions, rewards, next_states, dones, returns = b
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

                self.optimizer.zero_grad()
                # [batch_size x state_shape]
                logits = self.actor(states)
                dist = self.dist_fn(logits=logits)
                # [batch_size x 1]
                values = self.critic(states)
                a_loss = -(dist.log_prob(actions) * (returns - values).detach()).mean()
                vf_loss = F.mse_loss(returns, values)
                ent_loss = dist.entropy().mean()
                loss = a_loss + self._w_vf * vf_loss - self._w_ent * ent_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self._grad_clip)
                self.optimizer.step()



