import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class VanillaQNet(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units=(128, 128, 128), device='cpu'):
        super().__init__()
        self.device = device
        self.sequential_model = [
            nn.Linear(state_shape, hidden_units[0]),
            nn.ReLU()]
        for i in range(1, len(hidden_units)):
            self.sequential_model += [nn.Linear(hidden_units[i-1], hidden_units[i]), nn.ReLU()]
        self.fc_body = nn.Sequential(*self.sequential_model)
        self.output = nn.Linear(hidden_units[-1], action_shape)
        self.to(device)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        batch = state.shape[0]
        state = state.view(batch, -1)
        h = self.fc_body(state)
        q = self.output(h)
        return q, h

