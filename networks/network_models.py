import torch
from torch import nn
import torch.nn.functional as F
import gpytorch


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
        self.output = nn.Linear(hidden_units[-1], action_shape, bias=False)
        self.to(device)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        batch = state.shape[0]
        state = state.view(batch, -1)
        h = self.fc_body(state)
        q = self.output(h)
        return q, h


class NatureConvQNet(nn.Module):
    def __init__(self, action_dim):
        super(NatureConvQNet, self).__init__()
        self.feature_dim = 512
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, self.feature_dim)
        self.output = nn.Linear(self.feature_dim, action_dim)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        h = F.relu(self.fc4(y))
        q = self.output(h)
        return q, h


class StandardGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type='rbf'):
        super(StandardGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel_type == 'linear':
            self.covar_module = gpytorch.kernels.LinearKernel()
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)