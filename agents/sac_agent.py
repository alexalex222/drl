#######################################################################
# Haarnoja, T., Zhou, A., Abbeel, P. and Levine, S.,                  #
# 2018, July. Soft Actor-Critic: Off-Policy Maximum Entropy           #
# Deep Reinforcement Learning with a Stochastic Actor.                #
# In International Conference on Machine Learning (pp. 1861-1870).    #
#######################################################################

from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseAgent


class SoftActorCriticAgent(BaseAgent):
    def __init__(self,
                 value_net,
                 q_net1,
                 q_net2,
                 policy_net,
                 value_optimizer,
                 q_optimizer1,
                 q_optimizer2,
                 policy_optimizer,
                 config,
                 **kwargs):
        super().__init__(**kwargs)
        # value network V(s)
        self.value_net = value_net
        # target value network
        self.target_value_net = deepcopy(value_net)
        self.target_value_net.eval()
        # 1st q network q(s, a)
        self.q_net1 = q_net1
        # 2nd q network q(s, a)
        self.q_net2 = q_net2
        # policy network
        self.policy_net = policy_net
        # optimizer for value network
        self.value_optimizer = value_optimizer
        # optimizer for 1st q network
        self.q_optimizer1 = q_optimizer1
        # optimizer for 2nd q network
        self.q_optimizer2 = q_optimizer2
        # optimizer for policy network
        self.policy_optimizer = policy_optimizer
        # discount factor
        self.gamma = config['discount_factor']
        # device: cpu or gpu
        self.device = torch.device(config['device'])
        # number of completed steps
        self.steps_done = 0
        # configuration
        self.config = config

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config['target_network_mix']) +
                               param * self.config['target_network_mix'])

    def get_action_eval(self, state):
        with torch.no_grad():
            action = self.policy_net.get_action(state)
        return action

    def get_action(self, state):
        self.steps_done += 1
        with torch.no_grad():
            action = self.policy_net.get_action(state)
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

        predicted_q_values1, _ = self.q_net1(states, actions)
        predicted_q_values2, _ = self.q_net2(states, actions)
        predicted_values, _ = self.value_net(states)
        new_actions, log_probs, epsilons, means, log_stds = self.policy_net.evaluate(states)

        # Eq (8)
        with torch.no_grad():
            target_values, _ = self.target_value_net(next_states)
        target_q_values = rewards + (1 - dones) * self.gamma * target_values

        # Eq (7) and (9)
        q_value_loss1 = F.mse_loss(predicted_q_values1, target_q_values.detach())
        q_value_loss2 = F.mse_loss(predicted_q_values2, target_q_values.detach())

        # update q_net1
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.q_optimizer1.step()
        # update q_net2
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.q_optimizer2.step()

        # update value_net based on Eq (5) and (6)
        predicted_new_q_values = torch.min(self.q_net1(states, new_actions)[0], self.q_net2(states, new_actions)[0])
        target_value_func = predicted_new_q_values - log_probs
        value_loss = F.mse_loss(predicted_values, target_value_func.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # update the policy net
        policy_loss = (log_probs - predicted_new_q_values).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # soft update on target value net
        self.soft_update(self.target_value_net, self.value_net)

        return {'value_loss': value_loss.item(),
                'q1_loss': q_value_loss1.item(),
                'q2_loss': q_value_loss2.item(),
                'policy_loss': policy_loss.item()}

