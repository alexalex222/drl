import torch
from torch import nn
import torch.nn.functional as F
import gpytorch


def layer_init(layer, w_scale=1.0):
    nn.init.kaiming_normal_(layer.weight.data, mode='fan_in', nonlinearity='relu')
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


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
    def __init__(self, action_dim, device='cpu'):
        super(NatureConvQNet, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.output = layer_init(nn.Linear(self.feature_dim, action_dim))
        self.to(device)

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


class VanillaQNetContinuous(nn.Module):
    def __init__(self, state_shape, hidden_units=(128, 128, 128), device='cpu', init_w=3e-3):
        super(VanillaQNetContinuous, self).__init__()
        super().__init__()
        self.device = device
        self.sequential_model = [
            nn.Linear(state_shape, hidden_units[0]),
            nn.ReLU()]
        for i in range(1, len(hidden_units)):
            self.sequential_model += [nn.Linear(hidden_units[i - 1], hidden_units[i]), nn.ReLU()]
        self.fc_body = nn.Sequential(*self.sequential_model)
        self.output = nn.Linear(hidden_units[-1], 1)
        self.output.weight.data.uniform_(-init_w, init_w)
        self.output.bias.data.uniform_(-init_w, init_w)
        self.to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = x.to(self.device)
        h = self.fc_body(x)
        q = self.output(h)
        return q


class PolicyNetworkContinuous(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units=(128, 128, 128), device='cpu',
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetworkContinuous, self).__init__()
        self.device = device
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.sequential_model = [
            nn.Linear(state_shape, hidden_units[0]),
            nn.ReLU()]
        for i in range(1, len(hidden_units)):
            self.sequential_model += [nn.Linear(hidden_units[i - 1], hidden_units[i]), nn.ReLU()]
        self.fc_body = nn.Sequential(*self.sequential_model)

        self.mean_linear = nn.Linear(hidden_units[-1], action_shape)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_units[-1], action_shape)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.to(device)

    def forward(self, state):
        h = self.fc_body(state)

        mean = self.mean_linear(h)
        log_std = self.log_std_linear(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(self.device))
        log_prob = torch.distributions.Normal(mean, std).log_prob(mean + std * z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(0, 1)
        z = normal.sample().to(self.device)
        action = torch.tanh(mean + std * z)

        action = action.detach().cpu().numpy()
        return action[0]