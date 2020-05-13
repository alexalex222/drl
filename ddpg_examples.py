import argparse
import json
from datetime import datetime
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from networks.network_models import DeterministicActorCriticContinuous
from agents.ddpg_agent import DeepDeterministicPolicyGradientAgent
from utils.replay_buffers import BasicBuffer
from utils.trainer import off_policy_trainer
from utils.normalizer import NormalizedActions


def run_ddpg():
    parser = argparse.ArgumentParser(description="Arguments to choose environment")
    parser.add_argument('--env',
                        choices=['Pendulum-v0',
                                 'MountainCarContinuous-v0',
                                 'LunarLanderContinuous-v2'
                                 ],
                        help='Choose an environment')
    args = parser.parse_args()
    config_file = 'config_files/' + args.env + '_ddpg_feature.json'
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
    actor_critic = DeterministicActorCriticContinuous(action_shape=config['action_shape'],
                                                      state_shape=config['state_shape'],
                                                      actor_hidden_units=tuple(config['actor_hidden_units']),
                                                      critic_hidden_units=tuple(config['critic_hidden_units']),
                                                      device=config['device']
                                                      )
    actor_optim = torch.optim.Adam(actor_critic.actor_params, lr=config['actor_lr'])
    critic_optim = torch.optim.Adam(actor_critic.critic_params, lr=config['critic_lr'])

    agent = DeepDeterministicPolicyGradientAgent(actor_critic=actor_critic,
                                                 actor_optimizer=actor_optim,
                                                 critic_optimizer=critic_optim,
                                                 config=config)

    # log
    writer = SummaryWriter(config['logdir'] +
                           '/' + 'ddpg_' + config['task'] + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

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
    run_ddpg()
