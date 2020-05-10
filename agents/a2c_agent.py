import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseAgent


class A2CAgent(BaseAgent):
    def __init__(self,
                 actor,
                 critic,
                 optimizer,
                 config,
                 **kwargs):
        super().__init__(**kwargs)
        self.actor = actor
        self.critic = critic
        self._w_vf = config['value_loss_weight']
        self._w_ent = config['entropy_loss_weight']


