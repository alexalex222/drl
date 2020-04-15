import argparse
import json
from datetime import datetime
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from network_models import VanillaQNet
from dqn_agent import DQNAgent
from buffers import BasicBuffer
from trainer import off_policy_trainer

def run_dqn():
    parser = argparse.ArgumentParser(description="Arguments to choose environment")
    parser.add_argument('--env',
                        choices=['CartPole-v0',
                                 'Pendulum-v0',
                                 'Acrobot-v1',
                                 'MountainCar-v0',
                                 'MountainCarContinuous-v0'
                                 ],
                        help='Choose an environment')
    args = parser.parse_args()
    config_file = 'config_files/' + args.env + '_dqn_feature.json'
    config = json.load(open(config_file))

    env = gym.make(config['task'])
    config['state_shape'] = env.observation_space.shape[0] or env.observation_space.n
    config['action_shape'] = env.action_space.shape or env.action_space.n
    # model
    q_net = VanillaQNet(state_shape=config['state_shape'],
                        action_shape=config['action_shape'],
                        hidden_units=(128, 256, 256),
                        device=config['device'])
    optim = torch.optim.Adam(q_net.parameters(), lr=config['lr'])
    agent = DQNAgent(q_net=q_net,
                     optimizer=optim,
                     config=config)
    # log
    writer = SummaryWriter(config['logdir'] +
                           '/' + 'dqn_' + config['task'] + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    replay_buffer = BasicBuffer(max_size=config['buffer_size'])
    off_policy_trainer(env,
                       agent,
                       replay_buffer,
                       writer,
                       config['max_episodes'],
                       config['max_steps'],
                       config['batch_size'])

if __name__ == '__main__':
    run_dqn()
