# %%
import gym

env = gym.make('CartPole-v0')
state = env.reset()
i = 0
while True:
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    if done:
        break
    state = next_state
    i = i + 1
    print(i)


# %%
import numpy as np
import torch
from component import *
from network import *
from utils import *
from matplotlib import pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
task = Task('CartPole-v0')
i = 0
total_data = np.empty((0, 4), dtype=float)


for i_episode in range(50):
    print('Running Ep: ', i_episode)
    s = task.reset()
    current_episode_steps = 0
    while True:
        a = np.asarray([task.action_space.sample()])
        s_, r, d, info = task.step(a)
        current_episode_steps += 1

        one_record = np.hstack(s).reshape(1, -1)
        total_data = np.append(total_data, one_record, axis=0)

        if d:
            break
        s = s_


# %%
game = 'CartPole-v0'
select_device(0)
feature_size = 8
q_net = VanillaNetFeature(2, FCBody(state_dim=4, hidden_units=(128, 64, feature_size)))
q_net.load_state_dict(torch.load('data/{0}_q_net.pt'.format(game)))
states = torch.tensor(total_data, dtype=torch.float32).to(Config.DEVICE)

with torch.no_grad():
    q_value, features = q_net(states)


q_value = q_value.detach()
features = features.detach()


# %%
likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
likelihood1.initialize(noise=1e-4)
likelihood2.initialize(noise=1e-4)
gp1 = StandardGPModel(features[:1000, :], q_value[:1000, 0], likelihood1, kernel_type='linear')
gp2 = StandardGPModel(features[:1000, :], q_value[:1000, 1], likelihood2, kernel_type='linear')

gp3 = StandardGPModel(None, None, likelihood2, kernel_type='linear')

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
    q1_gp_dist = gp1(features[1000:, :])
    q2_gp_dist = gp2(features[1000:, :])
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
ax.plot(q_value[1000:, 0].cpu().numpy())
ax.plot(q1_gp)
#ax.plot(q1_gp_upper)
#ax.plot(q1_gp_lower)
plt.show()


fig, ax = plt.subplots()
ax.plot(q_value[1000:, 1].cpu().numpy())
ax.plot(q2_gp)
#ax.plot(q2_gp_upper)
#ax.plot(q2_gp_lower)
plt.show()