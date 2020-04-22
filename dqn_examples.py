import argparse
import json
from datetime import datetime
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from networks.network_models import VanillaQNet
from agents.dqn_agent import DQNAgent
from utils.replay_buffers import BasicBuffer
from utils.trainer import off_policy_trainer
from utils.normalizer import RescaleNormalizer


def run_dqn():
    parser = argparse.ArgumentParser(description="Arguments to choose environment")
    parser.add_argument('--env',
                        choices=['CartPole-v0',
                                 'CartPole-v1',
                                 'Acrobot-v1',
                                 'MountainCar-v0'
                                 ],
                        help='Choose an environment')
    args = parser.parse_args()
    config_file = 'config_files/' + args.env + '_dqn_feature.json'
    config = json.load(open(config_file))

    # make an environment
    env = gym.make(config['task'])
    # get state shape
    if isinstance(env.observation_space, gym.spaces.discrete.Discrete):
        config['state_shape'] = env.observation_space.n
    elif isinstance(env.observation_space, gym.spaces.box.Box):
        config['state_shape'] = env.observation_space.shape[0]

    # get action shape
    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        config['action_shape'] = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.box.Box):
        config['action_shape'] = env.action_space.shape[0]

    # model
    q_net = VanillaQNet(state_shape=config['state_shape'],
                        action_shape=config['action_shape'],
                        hidden_units=tuple(config['hidden_units']),
                        device=config['device'])
    # optimizer
    optim = torch.optim.Adam(q_net.parameters(), lr=config['lr'])
    # agent
    agent = DQNAgent(q_net=q_net,
                     optimizer=optim,
                     config=config)
    # log
    writer = SummaryWriter(config['logdir'] +
                           '/' + 'dqn_' + config['task'] + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # replay buffer
    replay_buffer = BasicBuffer(max_size=config['buffer_size'])
    # trainer
    off_policy_trainer(env,
                       agent,
                       replay_buffer,
                       writer,
                       config,
                       reward_normalizer=RescaleNormalizer(0.01)
                       )
    # save q_net
    torch.save(q_net.state_dict(), config['q_net_save_path'] + config['task'] + '_dqn' + '.pt')
    writer.close()

if __name__ == '__main__':
    run_dqn()
