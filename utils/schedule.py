# Adapted from https://github.com/ShangtongZhang/DeepRL.git

#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import math


class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val


class ExponentialSchedule:
    def __init__(self, start, end, eps_decay):
        self.start = start
        self.eps_decay = eps_decay
        self.current = start
        self.end = end
        self.current_step = 0
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self):
        val = self.current
        self.current_step += 1
        self.current = self.end + (self.start - self.end) * math.exp(-1. * self.current_step / self.eps_decay)
        return val


def epsilon_decay(eps, step, config):
    if config['type'] == 'exponential':
        return epsilon_decay_exp(
            eps=eps,
            min_eps=config['min_epsilon'],
            decay=config['decay_eps']
        )

    if config['type'] == 'expo_step':
        return epsilon_decay_exp_step(
            step=step,
            ini_eps=config['init_epsilon'],
            min_eps=config['min_epsilon'],
            lamda=0.001
        )

    if config['type'] == 'linear_anneal':
        return epsilon_linear_anneal(
            eps=eps,
            ini_eps=config['init_epsilon'],
            min_eps=config['min_epsilon'],
            timesteps=config['decay_steps']
        )


def epsilon_decay_exp(eps, min_eps, decay=0.99):
    return max(eps*decay, min_eps)


def epsilon_decay_exp_step(step, ini_eps, min_eps, lamda=0.001):
    return min_eps + (ini_eps - min_eps) * math.exp(-lamda * step)


def epsilon_linear_anneal(eps, ini_eps, min_eps, timesteps=10000):
    delta = (ini_eps - min_eps)/float(timesteps)
    return max(eps - delta, min_eps)
