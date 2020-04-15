import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter


def off_policy_trainer(env, agent, replay_buffer, writer, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            writer.add_scalar('Epsilon', agent.epsilon, agent.steps_done)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                batch_data = replay_buffer.sample(batch_size)
                results_dict = agent.learn(batch_data)

                writer.add_scalar('Q_Net_Loss', results_dict['q_net_loss'], agent.steps_done)
                writer.add_scalar('Epsilon', results_dict['eps'], agent.steps_done)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

        writer.add_scalar('Episode Reward', episode_reward, episode + 1)

    return episode_rewards