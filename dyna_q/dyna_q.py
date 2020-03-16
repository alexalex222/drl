import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import utils
from networks.network_bodies import FCBody
from networks.network_heads import VanillaNet, EnvModel
import gym


class DynaQ(object):
    def __init__(self, config):
        self.config = config
        self.epsilon = self.config['exploration']['init_epsilon']
        self.total_steps = 0
        self.current_episode_steps = 0
        self.env = gym.make(config['env_id'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.env_a_shape = 0 if isinstance(self.env.action_space.sample(), int) \
            else self.env.action_space.sample().shape
        self.Q_H1Size = 64
        self.Q_H2Size = 32
        self.env_H1Size = 64
        self.env_H2Size = 32
        self.eval_net = VanillaNet(self.n_actions, FCBody(self.n_states, hidden_units=(self.Q_H1Size, self.Q_H2Size)))
        self.target_net = deepcopy(self.eval_net)
        self.env_model = EnvModel(self.n_states, 1, self.env_H1Size, self.env_H2Size)
        self.eval_net = self.eval_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.env_model = self.env_model.to(self.device)
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((self.config['memory']['memory_capacity'], self.n_states * 2 + 3))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.config['learning_rate'])
        self.env_opt = torch.optim.Adam(self.env_model.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()

    # self.loss_func = nn.SmoothL1Loss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.tensor(x, dtype=torch.float32, device=self.device), 0)
        # input only one sample
        if np.random.uniform() > self.epsilon:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)  # return the argmax index
        else:  # random
            action = np.random.randint(0, self.n_actions)
            action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        return action

    def store_transition(self, s, a, r, s_, d):
        transition = np.hstack((s, [a, r, d], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.config['memory']['memory_capacity']
        self.memory[index, :] = transition
        self.memory_counter += 1

    def store_batch_transitions(self, experiences):
        index = self.memory_counter % self.config['memory']['memory_capacity']
        for exp in experiences:
            self.memory[index, :] = exp
            self.memory_counter += 1

    def update_env_model(self):
        sample_index = np.random.choice(min(self.config['memory']['memory_capacity'], self.memory_counter),
                                        self.config['batch_size'])
        b_memory = self.memory[sample_index, :]

        b_in = torch.tensor(np.hstack((b_memory[:, :self.n_states], b_memory[:, self.n_states:self.n_states + 1])),
                            dtype=torch.float32, device=self.device
                            )
        # b_y = Variable(torch.FloatTensor(np.hstack((b_memory[:, -self.n_states:], b_memory[:, self.n_states+1:self.n_states+2], b_memory[:, self.n_states+2:self.n_states+3]))))
        b_y_s = torch.tensor(b_memory[:, -self.n_states:], dtype=torch.float32, device=self.device)
        b_y_r = torch.tensor(b_memory[:, self.n_states + 1:self.n_states + 2], dtype=torch.float32, device=self.device)
        b_y_d = torch.tensor(b_memory[:, self.n_states + 2:self.n_states + 3], dtype=torch.float32, device=self.device)

        self.env_model.train()
        b_s_, b_r, b_d = self.env_model(b_in)
        # loss = self.loss_func(torch.cat(b_out, 1), b_y)
        loss_s = self.loss_func(b_s_, b_y_s)
        loss_r = self.loss_func(b_r, b_y_r)
        loss_d = self.loss_func(b_d, b_y_d)

        self.env_opt.zero_grad()
        loss_s.backward(retain_graph=True)
        self.env_opt.step()

        self.env_opt.zero_grad()
        loss_r.backward(retain_graph=True)
        self.env_opt.step()

        self.env_opt.zero_grad()
        loss_d.backward()
        self.env_opt.step()

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.config['target_update_freq'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(min(self.config['memory']['memory_capacity'], self.memory_counter),
                                        self.config['batch_size'])
        b_memory = self.memory[sample_index, :]
        b_s = torch.tensor(b_memory[:, :self.n_states], dtype=torch.float32, device=self.device)
        b_a = torch.tensor(b_memory[:, self.n_states:self.n_states + 1].astype(int), dtype=torch.long,
                           device=self.device)
        b_r = torch.tensor(b_memory[:, self.n_states + 1:self.n_states + 2], dtype=torch.float32, device=self.device)
        b_d = torch.tensor(1 - b_memory[:, self.n_states + 2:self.n_states + 3], dtype=torch.float32,
                           device=self.device)
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
            q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
            q_target = b_r + self.config['discount'] * q_next.max(1)[0].view(self.config['batch_size'],
                                                                             1) * b_d  # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(self.eval_net.parameters(), self.config['clipping_value'])
        # for param in self.eval_net.parameters():
        # 	param.grad.data.clamp_(-0.5, 0.5)
        self.optimizer.step()

    def simulate_learn(self):
        # target parameter update
        if self.learn_step_counter % self.config['target_update_freq'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(min(self.config['memory']['memory_capacity'], self.memory_counter),
                                        self.config['batch_size'])
        b_memory = self.memory[sample_index, :]
        b_s = b_memory[:, :self.n_states]

        # # cartpole random generated data
        # b_s_s = np.random.uniform(low=-2.4, high=2.4, size=(self.config['batch_size'], 1))
        # b_s_theta = np.random.uniform(low=-0.2094, high=0.2094, size=(self.config['batch_size'], 1))
        # b_s_v = np.random.normal(scale=10, size=(self.config['batch_size'], 1))
        # b_s_w = np.random.normal(scale=10, size=(self.config['batch_size'], 1))
        # b_s = np.hstack((b_s_s, b_s_v, b_s_theta, b_s_w))

        # mountaincar random generated data
        # b_s_s = np.random.uniform(low=-1.2, high=0.6, size=(self.config['batch_size'], 1))
        # b_s_v = np.random.uniform(low=-0.07, high=0.07, size=(self.config['batch_size'], 1))
        # b_s = np.hstack((b_s_s, b_s_v))

        b_a = np.random.randint(self.n_actions, size=b_s.shape[0])
        b_a = np.reshape(b_a, (b_a.shape[0], 1))
        b_in = torch.tensor(np.hstack((b_s, np.array(b_a))), dtype=torch.float32, device=self.device)

        self.env_model.eval()
        with torch.no_grad():
            state_prime_value, reward_value, done_value = self.env_model(b_in)

        # check if the episode is done
        # x, _, theta, _ = state_prime_value.cpu().data.numpy()
        x = state_prime_value[:, [0]]
        theta = state_prime_value[:, [2]]
        done_value = (torch.abs(x) > self.config['cart_position_limit']) | (torch.abs(theta) > self.config['pole_angle_limit'])

        b_s = torch.tensor(b_s, dtype=torch.float32, device=self.device)
        b_a = torch.tensor(b_a, dtype=torch.long, device=self.device)
        b_d = torch.tensor(1 - done_value.cpu().data.numpy(), dtype=torch.float32, device=self.device)
        b_s_ = torch.tensor(state_prime_value.cpu().data.numpy(), dtype=torch.float32, device=self.device)
        b_r = torch.tensor(reward_value.cpu().data.numpy(), dtype=torch.float32, device=self.device)

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
            q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
            q_target = b_r + self.config['discount'] * q_next.max(1)[0].view(self.config['batch_size'],
                                                                             1) * b_d  # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(self.eval_net.parameters(), self.config['clipping_value'])
        # for param in self.eval_net.parameters():
        # 	param.grad.data.clamp_(-0.5, 0.5)
        self.optimizer.step()

    def train_agent(self, max_episodes):
        rwd_dyna = []
        for i_episode in range(max_episodes):
            s = self.env.reset()
            ep_r = 0
            self.current_episode_steps = 0
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
                self.current_episode_steps += 1

                # start update policy when memory has enough exps
                if self.memory_counter > self.config['first_update']:
                    self.learn()

                # start update env model when memory has enough exps
                if self.memory_counter > self.config['batch_size']:
                    self.update_env_model()

                # planning through generated exps
                for _ in range(self.config['simulation_rounds']):
                    self.simulate_learn()

                if done:
                    print('Dyna-Q | Ep: ', i_episode + 1, '| timestep: ', self.current_episode_steps, '| Ep_r: ', ep_r)
                    rwd_dyna.append(ep_r)
                    break
                s = s_

        return rwd_dyna
