import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseAgent


class A2CAgent(BaseAgent):
    def __init__(self,
                 actor,
                 critic,
                 actor_optimizer,
                 critic_optimizer,
                 env,
                 config,
                 **kwargs):
        super().__init__(**kwargs)
        self.actor = actor
        self.critic = critic
        self._w_vf = config['value_loss_weight']
        self._w_ent = config['entropy_loss_weight']
        self.gamma = config['discount_factor']
        self.steps_done = 0
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        # device: cpu or gpu
        self.device = torch.device(config['device'])
        self.config = config

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), [next_value], axis=0)

        for t in reversed(range(self.config['roll_out_length'])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def get_action(self, state):
        self.steps_done += 1
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)


