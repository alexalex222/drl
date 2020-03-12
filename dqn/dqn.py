import torch
import torch.nn as nn
import gym
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import utils
from utils.replay_buffers import PriorExpReplay
from networks.network_heads import VanillaNet
from networks.network_bodies import FCBody


class Q_Net(nn.Module):
	def __init__(self, N_STATES, N_ACTIONS, H1Size, H2Size):
		super(Q_Net, self).__init__()
		# build network layers
		self.fc1 = nn.Linear(N_STATES, H1Size)
		self.fc2 = nn.Linear(H1Size, H2Size)
		self.out = nn.Linear(H2Size, N_ACTIONS)

		# initialize layers
		utils.torch_utils.weights_init_normal([self.fc1, self.fc2, self.out], 0.0, 0.1)
		#utils.weights_init_xavier([self.fc1, self.fc2, self.out], False)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		actions_value = self.out(x)

		return actions_value


class DQN(object):
	def __init__(self, config):
		self.config = config
		self.epsilon = self.config['exploration']['init_epsilon']
		self.total_steps = 0
		self.env = gym.make(config['env_id'])
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.n_states = self.env.observation_space.shape[0]
		self.n_actions = self.env.action_space.n
		self.env_a_shape = 0 if isinstance(self.env.action_space.sample(), int) \
			else self.env.action_space.sample().shape
		self.H1Size = 64
		self.H2Size = 32
		# self.eval_net = Q_Net(self.n_states, self.n_actions, self.H1Size, self.H2Size)
		self.eval_net = VanillaNet(self.n_actions, FCBody(self.n_states, hidden_units=(self.H1Size, self.H2Size)))
		self.target_net = deepcopy(self.eval_net)
		self.eval_net = self.eval_net.to(self.device)
		self.target_net = self.target_net.to(self.device)
		self.learn_step_counter = 0                                     # for target updating
		self.memory_counter = 0                                         # for storing memory
		if self.config['memory']['prioritized']:
			self.memory = PriorExpReplay(self.config['memory']['memory_capacity'])
		else:
			self.memory = np.zeros((self.config['memory']['memory_capacity'], self.n_states * 2 + 3))     # initialize memory
		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.config['learning_rate'])
		self.mse_loss = nn.MSELoss()
		self.mse_element_loss = nn.MSELoss(reduce=False)
		self.l1_loss = nn.L1Loss(reduce=False)
		#self.mse_loss = nn.SmoothL1Loss()

	def choose_action(self, x):
		x = torch.unsqueeze(torch.tensor(x, dtype=torch.float32, device=self.device), 0)
		# input only one sample
		if np.random.uniform() > self.epsilon:   # greedy
			actions_value = self.eval_net.forward(x)
			action = torch.max(actions_value, 1)[1].cpu().data.numpy()
			action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)  # return the argmax index
		else:   # random
			action = np.random.randint(0, self.n_actions)
			action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
		return action

	def store_transition_per(self, s, a, r, s_, d):
		transition = np.hstack((s, [a, r, d], s_))
		error = self.get_TD_error(transition.reshape(1, transition.shape[0]))
		# replace the old memory with new memory
		self.memory_per.add(error, transition)

	def store_transition(self, s, a, r, s_, d):
		transition = np.hstack((s, [a, r, d], s_))
		if self.config['memory']['prioritized']:
			self.memory.store(transition)
		else:
			# replace the old memory with new memory
			index = self.memory_counter % self.config['memory']['memory_capacity']
			self.memory[index, :] = transition
		self.memory_counter += 1

	def store_batch_transitions(self, experiences):
		index = self.memory_counter % self.config['memory']['memory_capacity']
		for exp in experiences:
			self.memory[index, :] = exp
			self.memory_counter += 1

	def clear_memory(self):
		self.memory_counter = 0
		self.memory = np.zeros((self.config['memory']['memory_capacity'], self.n_states * 2 + 3))

	def get_TD_error(self, transition):
		b_s = torch.tensor(transition[:, :self.n_states], dtype=torch.float32)
		b_a = torch.tensor(transition[:, self.n_states:self.n_states+1].astype(int), dtype=torch.float32)
		b_r = transition[:, self.n_states+1:self.n_states+2]
		b_d = 1 - transition[:, self.n_states+2:self.n_states+3]
		b_s_ = torch.tensor(transition[:, -self.n_states:], dtype=torch.float32)

		# q_eval w.r.t the action in experience
		q_eval = self.eval_net(b_s).gather(1, b_a)
		q_eval_next = self.eval_net(b_s_)
		q_argmax = np.argmax(q_eval_next.data.numpy(), axis=1)
		q_next = self.target_net(b_s_)
		q_next_numpy = q_next.data.numpy()
		q_update = q_next_numpy[0, q_argmax[0]]
		q_target = b_r + self.config['discount'] * q_update * b_d

		return abs(q_eval.data.numpy()[0] - q_target[0][0])

	def learn(self):
		# target parameter update
		if self.learn_step_counter % self.config['target_update_freq'] == 0:
			self.target_net.load_state_dict(self.eval_net.state_dict())
		self.learn_step_counter += 1

		# sample batch transitions
		if self.config['memory']['prioritized']:
			tree_idx, b_memory, ISWeights = self.memory.sample(self.config['batch_size'], self.memory_counter)
			b_weights = torch.tensor(ISWeights, dtype=torch.float32)
		else:
			sample_index = np.random.choice(min(self.config['memory']['memory_capacity'], self.memory_counter), self.config['batch_size'])
			b_memory = self.memory[sample_index, :]

		b_s = torch.tensor(b_memory[:, :self.n_states], dtype=torch.float32, device=self.device)
		b_a = torch.tensor(b_memory[:, self.n_states:self.n_states+1].astype(int), dtype=torch.long, device=self.device)
		b_r = torch.tensor(b_memory[:, self.n_states+1:self.n_states+2], dtype=torch.float32, device=self.device)
		b_d = torch.tensor(1 - b_memory[:, self.n_states+2:self.n_states+3], dtype=torch.float32, device=self.device)
		b_s_ = torch.tensor(b_memory[:, -self.n_states:], dtype=torch.float32, device=self.device)

		# q_eval w.r.t the action in experience
		q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
		
		if self.config['double_q_model']:
			q_eval_next = self.eval_net(b_s_)
			q_argmax = np.argmax(q_eval_next.cpu().data.numpy(), axis=1)
			q_next = self.target_net(b_s_)
			q_next_numpy = q_next.cpu().data.numpy()
			q_update = np.zeros((self.config['batch_size'], 1))
			for i in range(self.config['batch_size']):
				q_update[i] = q_next_numpy[i, q_argmax[i]]
			q_target = b_r + torch.tensor(self.config['discount'] * q_update,
										  dtype=torch.float32, device=self.device) * b_d
		else:
			q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
			q_target = b_r + self.config['discount'] * q_next.max(1)[0].view(self.config['batch_size'], 1) * b_d  # shape (batch, 1)

		if self.config['memory']['prioritized']:
			abs_errors = self.l1_loss(q_eval, q_target)
			loss = (b_weights*self.mse_element_loss(q_eval, q_target)).mean()
			self.memory.batch_update(tree_idx, abs_errors.data.numpy())
		else:
			loss = self.mse_loss(q_eval, q_target)

		self.optimizer.zero_grad()
		loss.backward()
		# nn.utils.clip_grad_norm(self.eval_net.parameters(), self.config['clipping_value'])
		# for param in self.eval_net.parameters():
		# 	param.grad.data.clamp_(-1.0, 1.0)
		self.optimizer.step()

		return loss.item()

	def train_agent(self, max_episodes):
		rwd_dqn = []
		for i_episode in range(max_episodes):
			s = self.env.reset()
			ep_r = 0
			timestep = 0
			while True:
				self.total_steps += 1
				# decay exploration
				self.epsilon = utils.schedule.epsilon_decay(
					eps=self.epsilon,
					step=self.total_steps,
					config=self.config['exploration']
				)

				# env.render()
				a = self.choose_action(s)

				# take action
				s_, r, done, info = self.env.step(a)
				ep_r += r

				# modify the reward
				if self.config['modify_reward']:
					r = utils.normalizer.modify_rwd(self.config['env_id'], s_)

				# store current transition
				self.store_transition(s, a, r, s_, done)
				timestep += 1

				# start update policy when memory has enough exps
				if self.memory_counter > self.config['first_update']:
					self.learn()

				if done:
					prefix = 'DDQN' if self.config['double_q_model'] else 'DQN'
					if self.config['memory']['prioritized']:
						prefix += '-PER'
					print(prefix + '| Ep: ', i_episode + 1, '| timestep: ', timestep, '| Ep_r: ', ep_r)
					rwd_dqn.append(ep_r)
					break

				s = s_
		return rwd_dqn
