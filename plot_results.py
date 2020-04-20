# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 200
plt.style.use('ggplot')

# %%
task = 'CartPole'
min_reward = 0
max_reward = 200
dqn_rewards = pd.read_csv('results/logs/dqn_{0}.csv'.format(task), sep=',', header=None).values
dqn_rewards_mean = dqn_rewards.mean(axis=1)
dqn_rewards_std = dqn_rewards.std(axis=1)
dqn_rewards_upper = np.clip(dqn_rewards_mean + dqn_rewards_std, min_reward, max_reward)
dqn_rewards_lower = np.clip(dqn_rewards_mean - dqn_rewards_std, min_reward, max_reward)


fig, ax = plt.subplots()
ax.plot(dqn_rewards_mean,
        color="#0072B2", label='DQN')
ax.fill_between(np.arange(len(dqn_rewards_mean)),
                dqn_rewards_upper,
                dqn_rewards_lower,
                facecolor="#0072B2", alpha=0.2)

ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
#ax.set_xlim([0, 500])
ax.set_title(task)
plt.tight_layout()
plt.show()
