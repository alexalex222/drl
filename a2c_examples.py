import argparse
import json
from datetime import datetime
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from networks.network_models import MLPActor, MLPCritic
from agents.a2c_agent import AdvantageActorCriticAgent
from utils.replay_buffers import BasicBuffer
from utils.trainer import a2c_policy_trainer
from utils.normalizer import RescaleNormalizer


def run_a2c():
    parser = argparse.ArgumentParser(description="Arguments to choose environment")
    parser.add_argument('--env',
                        choices=['CartPole-v0',
                                 'CartPole-v1',
                                 'Acrobot-v1',
                                 'MountainCar-v0'
                                 ],
                        help='Choose an environment')
    args = parser.parse_args()
    config_file = 'config_files/' + args.env + '_a2c_feature.json'
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
        dist_fn = torch.distributions.Categorical
    elif isinstance(env.action_space, gym.spaces.box.Box):
        config['action_shape'] = env.action_space.shape[0]
        dist_fn = torch.distributions.Normal

    # model
    critic = MLPCritic(state_shape=config['state_shape'],
                       hidden_units=tuple(config['hidden_units']),
                       device=config['device'])
    actor = MLPActor(state_shape=config['state_shape'],
                     action_shape=config['action_shape'],
                     hidden_units=tuple(config['hidden_units']),
                     device=config['device'])
    # optimizer
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=config['lr'])
    # agent
    agent = AdvantageActorCriticAgent(actor=actor,
                                      critic=critic,
                                      optimizer=optim,
                                      dist_fn=dist_fn,
                                      config=config)
    # log
    writer = SummaryWriter(config['logdir'] +
                           '/' + 'a2c_' + config['task'] + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # replay buffer
    replay_buffer = BasicBuffer(max_size=config['buffer_size'])
    # trainer
    a2c_policy_trainer(env,
                       agent,
                       replay_buffer,
                       writer,
                       config,
                       reward_normalizer=RescaleNormalizer(0.01)
                       )
    # save q_net
    # torch.save(q_net.state_dict(), config['q_net_save_path'] + config['task'] + '_a2c' + '.pt')
    writer.close()


if __name__ == '__main__':
    run_a2c()
