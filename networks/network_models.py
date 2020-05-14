import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gpytorch


def layer_init(layer, w_scale=1.0):
    nn.init.kaiming_normal_(layer.weight.data, mode='fan_in', nonlinearity='relu')
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x


class MLPQNet(nn.Module):
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


class MLPCritic(nn.Module):
    def __init__(self, state_shape, hidden_units=(128, 128, 128), device='cpu'):
        super().__init__()
        self.device = device
        self.sequential_model = [
            nn.Linear(state_shape, hidden_units[0]),
            nn.ReLU()]
        for i in range(1, len(hidden_units)):
            self.sequential_model += [nn.Linear(hidden_units[i-1], hidden_units[i]), nn.ReLU()]
        self.fc_body = nn.Sequential(*self.sequential_model)
        self.output = nn.Linear(hidden_units[-1], 1)
        self.to(device)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        batch = state.shape[0]
        state = state.view(batch, -1)
        h = self.fc_body(state)
        v = self.output(h)
        return v


class MLPCategoricalActor(nn.Module):
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
        logits = self.output(h)
        return logits


class MLPActorContinuousDeterministic(nn.Module):
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
        self.output.weight.data.uniform_(-1e-3, 1e-3)
        self.output.bias.data.uniform_(-1e-3, 1e-3)
        self.to(device)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        batch = state.shape[0]
        state = state.view(batch, -1)
        h = self.output(self.fc_body(state))
        action = torch.tanh(h)
        return action


class MLPGaussianActor(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units=(128, 128, 128), device='cpu'):
        super().__init__()
        self.device = device
        self.sequential_model = [
            nn.Linear(state_shape, hidden_units[0]),
            nn.ReLU()]
        for i in range(1, len(hidden_units)):
            self.sequential_model += [nn.Linear(hidden_units[i - 1], hidden_units[i]), nn.ReLU()]
        self.fc_body = nn.Sequential(*self.sequential_model)
        self.mean_linear = nn.Linear(hidden_units[-1], action_shape)
        self.mean_linear.weight.data.uniform_(-1e-3, 1e-3)
        self.mean_linear.bias.data.uniform_(-1e-3, 1e-3)
        # we use homoscedastic Gaussian noise
        log_std = -0.5 * np.ones(action_shape, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std), requires_grad=True)
        self.to(device)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        batch = state.shape[0]
        # reshape the feature to [batch_size x state_shape]
        state = state.view(batch, -1)
        h = self.fc_body(state)
        mean = self.mean_linear(h)
        std = torch.exp(self.log_std)
        return mean, std


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


class MLPQNetContinuous(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units=(128, 128, 128), device='cpu', init_w=3e-3):
        super().__init__()
        self.device = device
        self.sequential_model = [
            nn.Linear(state_shape + action_shape, hidden_units[0]),
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
        return q, h


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
        log_prob = torch.unsqueeze(torch.sum(log_prob, 1), 1)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(0, 1)
        z = normal.sample().to(self.device)
        action = torch.tanh(mean + std * z)
        action = action.squeeze(0)
        action = action.detach().cpu().numpy()
        return action


class CategoricalActorCriticNet(nn.Module):
    def __init__(self,
                 state_shape,
                 action_shape,
                 hidden_units=(128, 128, 128),
                 device='cpu'):
        super(CategoricalActorCriticNet, self).__init__()
        self.device = device
        self.sequential_model = [
            nn.Linear(state_shape, hidden_units[0]),
            nn.ReLU()]
        for i in range(1, len(hidden_units)):
            self.sequential_model += [nn.Linear(hidden_units[i - 1], hidden_units[i]), nn.ReLU()]
        self.fc_body = nn.Sequential(*self.sequential_model)
        # both actor and critic share a common feature extractor
        # actor computes the logits for actions
        self.actor = nn.Linear(hidden_units[-1], action_shape)
        self.actor.weight.data.uniform_(-1e-3, 1e-3)
        self.actor.bias.data.uniform_(-1e-3, 1e-3)
        # critic computes V(s)
        self.critic = nn.Linear(hidden_units[-1], 1)
        self.critic.weight.data.uniform_(-1e-3, 1e-3)
        self.critic.bias.data.uniform_(-1e-3, 1e-3)
        self.to(device)

    def forward(self, state, action=None):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        batch = state.shape[0]
        # reshape the feature to [batch_size x state_shape]
        state = state.view(batch, -1)
        feature = self.fc_body(state)
        logits = self.actor(feature)
        value = self.critic(feature)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return action, log_prob, entropy, value


class DeterministicActorCriticContinuous(nn.Module):
    def __init__(self,
                 state_shape,
                 action_shape,
                 actor_hidden_units=(128, 128, 128),
                 critic_hidden_units=(128, 128, 128),
                 feature_extractor=None,
                 device='cpu'):
        super().__init__()
        self.device = device
        if feature_extractor is None:
            feature_extractor = DummyBody(state_dim=state_shape)
        self.feature_extractor = feature_extractor
        # actor compute a = \mu(s)
        self.actor = MLPActorContinuousDeterministic(state_shape=feature_extractor.feature_dim,
                                                     action_shape=action_shape,
                                                     hidden_units=actor_hidden_units,
                                                     device=device)
        # critic computes q(s, a)
        self.critic = MLPQNetContinuous(state_shape=feature_extractor.feature_dim,
                                        action_shape=action_shape,
                                        hidden_units=critic_hidden_units,
                                        device=device)

        # actor parameters
        self.actor_params = list(self.actor.parameters()) + list(self.feature_extractor.parameters())
        # critic parameters
        self.critic_params = list(self.critic.parameters()) + list(self.feature_extractor.parameters())
        self.to(device)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        phi = self.feature_extractor(state)
        action = self.actor(phi)
        return action

    def act(self, phi):
        action = self.actor(phi)
        return action

    def criticize(self, phi, action):
        q, _ = self.critic(phi, action)
        return q


class GaussianActorCriticContinuous(nn.Module):
    def __init__(self,
                 state_shape,
                 action_shape,
                 actor_hidden_units=(128, 128, 128),
                 critic_hidden_units=(128, 128, 128),
                 feature_extractor=None,
                 device='cpu'):
        super().__init__()
        self.device = device
        if feature_extractor is None:
            feature_extractor = DummyBody(state_dim=state_shape)
        self.feature_extractor = feature_extractor
        # actor compute a = \mu(s)
        self.actor = MLPActorContinuousDeterministic(state_shape=feature_extractor.feature_dim,
                                                     action_shape=action_shape,
                                                     hidden_units=actor_hidden_units,
                                                     device=device)
        # critic computes q(s, a)
        self.critic = MLPQNetContinuous(state_shape=feature_extractor.feature_dim,
                                        action_shape=action_shape,
                                        hidden_units=critic_hidden_units,
                                        device=device)

        # actor parameters
        self.actor_params = list(self.actor.parameters()) + list(self.feature_extractor.parameters())
        # critic parameters
        self.critic_params = list(self.critic.parameters()) + list(self.feature_extractor.parameters())
        self.to(device)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        phi = self.feature_extractor(state)
        action = self.actor(phi)
        return action

    def act(self, phi):
        action = self.actor(phi)
        return action

    def criticize(self, phi, action):
        q, _ = self.critic(phi, action)
        return q