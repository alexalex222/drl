#######################################################################
# Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T.,      #
# Tassa, Y., Silver, D. and Wierstra, D.,   2016, July.               #
#  Continuous control with deep reinforcement learning                #
# 4th International Conference on Learning Representations, 2016      #
#######################################################################

from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseAgent


class DeepDeterministicPolicyGradientAgent(BaseAgent):
    def __init__(self, actor_critic, actor_optimizer, critic_optimizer, config, **kwargs):
        super().__init__(**kwargs)
        self.actor_critic = actor_critic
        self.target_actor_critic = deepcopy(actor_critic)
        self.target_actor_critic.eval()
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.steps_done = 0
        # discount factor
        self.gamma = config['discount_factor']
        # Gaussian noise for exploration
        self.eps = config['exploration_noise']
        self.device = config['device']
        self.config = config

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config['target_network_mix']) +
                               param * self.config['target_network_mix'])

    def get_action(self, state):
        self.steps_done += 1
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor_critic(state).detach().cpu()
        action = action + torch.randn(size=action.shape) * self.eps
        action = action.numpy()[0]
        action = np.clip(action, -1, 1)
        return action

    def get_action_eval(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor_critic(state).detach().cpu()
        action = action.numpy()[0]
        return action

    def learn(self, batch, **kwargs):
        states, actions, rewards, next_states, dones = batch
        # states: shape [batch_size x state_dim]
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        # actions: shape [batch_size x action_dim]
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        # rewards: shape [batch_size x 1]
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        # next_states: shape [batch_size x state_dim]
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        # dones: shape [states x 1]
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            phis_next = self.target_actor_critic.feature_extractor(next_states)
            actions_next = self.target_actor_critic.act(phis_next)
            qs_next = self.target_actor_critic.criticize(phis_next, actions_next)
        qs_target = rewards + self.gamma * (1 - dones) * qs_next
        qs_target = qs_target.detach()

        phis = self.actor_critic.feature_extractor(states)
        qs = self.actor_critic.criticize(phis, actions)

        critic_loss = F.mse_loss(qs, qs_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        phis = self.actor_critic.feature_extractor(states)
        actions = self.actor_critic.act(phis)
        policy_loss = - self.actor_critic.criticize(phis.detach(), actions).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.target_actor_critic, self.actor_critic)

        return {'policy_loss': policy_loss.item(), 'critic_loss': critic_loss.item()}
