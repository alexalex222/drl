import argparse
import json
from datetime import datetime
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from networks.network_models import MLPCategoricalActor, MLPGaussianActor, MLPCritic
from agents.ppo_agent import ProximalPolicyOptimizationAgent
from utils.replay_buffers import BasicBuffer
from utils.trainer import on_policy_trainer
from utils.normalizer import RescaleNormalizer, NormalizedActions, MeanStdNormalizer


def run_ppo_discrete():
    parser = argparse.ArgumentParser(description="Arguments to choose environment")
    parser.add_argument('--env',
                        choices=['CartPole-v0',
                                 'CartPole-v1',
                                 'Acrobot-v1',
                                 'MountainCar-v0',
                                 ],
                        help='Choose an environment')
    args = parser.parse_args()
    config_file = 'config_files/' + args.env + '_ppo_feature.json'
    config = json.load(open(config_file))

    # make an environment
    env = gym.make(config['task'])
    # get state shape
    assert isinstance(env.observation_space, gym.spaces.discrete.Discrete), "Action Space should be discrete"

    config['state_shape'] = env.observation_space.n
    config['action_shape'] = env.action_space.n
    actor = MLPCategoricalActor(state_shape=config['state_shape'],
                                action_shape=config['action_shape'],
                                hidden_units=tuple(config['actor_hidden_units']),
                                device=config['device'])
    dist_fn = torch.distributions.Categorical

    # model
    critic = MLPCritic(state_shape=config['state_shape'],
                       hidden_units=tuple(config['critic_hidden_units']),
                       device=config['device'])

    # optimizer
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=config['lr'])
    # agent
    agent = ProximalPolicyOptimizationAgent(actor=actor,
                                            critic=critic,
                                            optimizer=optim,
                                            dist_fn=dist_fn,
                                            config=config)
    # log
    writer = SummaryWriter(config['logdir'] +
                           '/' + 'ppo_' + config['task'] + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # replay buffer
    replay_buffer = BasicBuffer(max_size=config['buffer_size'])
    # trainer
    on_policy_trainer(env,
                      agent,
                      replay_buffer,
                      writer,
                      config,
                      reward_normalizer=None
                      )
    # save q_net
    # torch.save(q_net.state_dict(), config['q_net_save_path'] + config['task'] + '_ppo' + '.pt')
    writer.close()


def run_ppo_continuous():
    parser = argparse.ArgumentParser(description="Arguments to choose environment")
    parser.add_argument('--env',
                        choices=[
                                 'Pendulum-v0',
                                 'MountainCarContinuous-v0'
                                 ],
                        help='Choose an environment')
    args = parser.parse_args()
    config_file = 'config_files/' + args.env + '_ppo_feature.json'
    config = json.load(open(config_file))

    # make an environment
    env = NormalizedActions(gym.make(config['task']))
    # get state shape
    config['state_shape'] = env.observation_space.shape[0]

    # get action shape
    config['action_shape'] = env.action_space.shape[0]
    actor = MLPGaussianActor(state_shape=config['state_shape'],
                             action_shape=config['action_shape'],
                             hidden_units=tuple(config['actor_hidden_units']),
                             device=config['device'])
    dist_fn = torch.distributions.Normal

    # model
    critic = MLPCritic(state_shape=config['state_shape'],
                       hidden_units=tuple(config['critic_hidden_units']),
                       device=config['device'])

    # optimizer
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=config['lr'], eps=1e-5)
    # agent
    agent = ProximalPolicyOptimizationAgent(actor=actor,
                                            critic=critic,
                                            optimizer=optim,
                                            dist_fn=dist_fn,
                                            config=config)
    # log
    writer = SummaryWriter(config['logdir'] +
                           '/' + 'ppo_' + config['task'] + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # replay buffer
    replay_buffer = BasicBuffer(max_size=config['buffer_size'])
    # trainer
    on_policy_trainer(env,
                      agent,
                      replay_buffer,
                      writer,
                      config,
                      state_normalizer=None,
                      reward_normalizer=None
                      )
    # save q_net
    # torch.save(q_net.state_dict(), config['q_net_save_path'] + config['task'] + '_ppo' + '.pt')
    writer.close()


if __name__ == '__main__':
    # run_ppo_discrete()
    run_ppo_continuous()
