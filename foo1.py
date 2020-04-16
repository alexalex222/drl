# %%
import numpy as np
import gym
import torch
import gpytorch
from networks.network_models import VanillaQNet, StandardGPModel
from matplotlib import pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
task = env = gym.make('CartPole-v0')
i = 0
total_data = np.empty((0, 4), dtype=float)


for i_episode in range(50):
    print('Running Ep: ', i_episode)
    s = task.reset()
    current_episode_steps = 0
    while True:
        a = task.action_space.sample()
        s_, r, d, info = task.step(a)
        current_episode_steps += 1

        one_record = np.hstack(s).reshape(1, -1)
        total_data = np.append(total_data, one_record, axis=0)

        if d:
            break
        s = s_


# %%
game = 'CartPole-v0'
q_net = VanillaQNet(state_shape=4,
                    action_shape=2,
                    hidden_units=[128, 128, 16],
                    device=device)
q_net.load_state_dict(torch.load('results/q_net_{0}_dqn.pt'.format(game)))
states = torch.tensor(total_data, dtype=torch.float32).to(device)

with torch.no_grad():
    q_value, features = q_net(states)


q_value = q_value.detach()
features = features.detach()

x_train_1 = features[:200, :].clone()
y_train_1 = q_value[:200, 0].clone()
x_test_1 = features[1000:, :].clone()
x_train_2 = features[:200, :].clone()
y_train_2 = q_value[:200, 1].clone()
x_test_2 = features[1000:, :].clone()

# %%
likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
likelihood1.initialize(noise=9e-4)
likelihood2.initialize(noise=9e-4)
gp1 = StandardGPModel(x_train_1, y_train_1, likelihood1, kernel_type='linear')
gp2 = StandardGPModel(x_train_2, y_train_2, likelihood2, kernel_type='linear')


gp1 = gp1.to(device)
gp2 = gp2.to(device)
likelihood1.to(device)
likelihood2.to(device)

# %%
gp1.eval()
gp2.eval()
likelihood1.eval()
likelihood2.eval()

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make predictions
    q1_gp_dist = gp1(x_test_1)
    q2_gp_dist = gp2(x_test_2)
    # Get upper and lower confidence bounds
    q1_gp = q1_gp_dist.mean.detach().cpu().numpy()
    q2_gp = q2_gp_dist.mean.detach().cpu().numpy()
    q1_gp_lower, q1_gp_upper = q1_gp_dist.confidence_region()
    q2_gp_lower, q2_gp_upper = q2_gp_dist.confidence_region()

    q1_gp_upper = q1_gp_upper.detach().cpu().numpy()
    q1_gp_lower = q1_gp_lower.detach().cpu().numpy()
    q2_gp_upper = q2_gp_upper.detach().cpu().numpy()
    q2_gp_lower = q2_gp_lower.detach().cpu().numpy()


# %%



fig, ax = plt.subplots()
ax.fill_between(np.arange(len(q1_gp)), q1_gp_lower, q1_gp_upper, facecolor='red', alpha=0.5)
ax.fill_between(np.arange(len(q2_gp)), q2_gp_lower, q2_gp_upper, facecolor='green', alpha=0.5)
plt.show()


fig, ax = plt.subplots()
ax.plot(q_value[1000:, 1].cpu().numpy())
ax.plot(q2_gp)
#ax.plot(q2_gp_upper)
#ax.plot(q2_gp_lower)
plt.show()


# %%
for param in q_net.output.parameters():
    print(param)