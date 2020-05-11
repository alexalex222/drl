import argparse
import json
from datetime import datetime
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from networks.network_models import VanillaQNet, VanillaQNetContinuous, PolicyNetworkContinuous
from agents.sac_agent import SoftActorCriticAgent
from utils.replay_buffers import BasicBuffer
from utils.trainer import off_policy_trainer
from utils.normalizer import NormalizedActions


def run_sac():
    parser = argparse.ArgumentParser(description="Arguments to choose environment")
    parser.add_argument('--env',
                        choices=['Pendulum-v0',
                                 'MountainCarContinuous-v0',
                                 ],
                        help='Choose an environment')
    args = parser.parse_args()
    config_file = 'config_files/' + args.env + '_sac_feature.json'
    config = json.load(open(config_file))

    # make an environment
    env = NormalizedActions(gym.make(config['task']))
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

    # value network
    value_net = VanillaQNet(state_shape=config['state_shape'],
                            action_shape=1,
                            hidden_units=tuple(config['value_net_hidden_units']),
                            device=config['device'])
    q_net1 = VanillaQNetContinuous(state_shape=config['state_shape'],
                                   action_shape=config['action_shape'],
                                   hidden_units=tuple(config['q_net_hidden_units']),
                                   device=config['device'])
    q_net2 = VanillaQNetContinuous(state_shape=config['state_shape'],
                                   action_shape=config['action_shape'],
                                   hidden_units=tuple(config['q_net_hidden_units']),
                                   device=config['device'])
    policy_net = PolicyNetworkContinuous(state_shape=config['state_shape'],
                                         action_shape=config['action_shape'],
                                         hidden_units=tuple(config['policy_net_hidden_units']),
                                         device=config['device']
                                         )
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=config['value_net_lr'])
    q_optimizer1 = torch.optim.Adam(q_net1.parameters(), lr=config['q_net_lr'])
    q_optimizer2 = torch.optim.Adam(q_net2.parameters(), lr=config['q_net_lr'])
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=config['policy_net_lr'])

    agent = SoftActorCriticAgent(value_net=value_net,
                                 q_net1=q_net1,
                                 q_net2=q_net2,
                                 policy_net=policy_net,
                                 value_optimizer=value_optimizer,
                                 q_optimizer1=q_optimizer1,
                                 q_optimizer2=q_optimizer2,
                                 policy_optimizer=policy_optimizer,
                                 config=config)

    # log
    writer = SummaryWriter(config['logdir'] +
                           '/' + 'sac_' + config['task'] + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # replay buffer
    replay_buffer = BasicBuffer(max_size=config['buffer_size'])
    # trainer
    off_policy_trainer(env,
                       agent,
                       replay_buffer,
                       writer,
                       config,
                       reward_normalizer=None
                       )


if __name__ == '__main__':
    run_sac()
