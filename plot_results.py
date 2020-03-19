# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 200
plt.style.use('ggplot')

# %%
env_id = 'AcroBot-v1'
gp_dyna_results = pd.read_csv('results/logs/gp_dyna_q_{}_run_5.csv'.format(env_id))
ddqn_results = pd.read_csv('results/logs/ddqn_{}_run_5.csv'.format(env_id))

fig, ax = plt.subplots()
ax.plot(gp_dyna_results['episodes'], gp_dyna_results['rewards'],
        color="#0072B2", label='GP Acceleration')
ax.fill_between(gp_dyna_results['episodes'],
                gp_dyna_results['rewards'] + 2*np.sqrt(gp_dyna_results['variances']),
                gp_dyna_results['rewards'] - 2*np.sqrt(gp_dyna_results['variances']),
                facecolor="#0072B2", alpha=0.2)
ax.plot(ddqn_results['episodes'], ddqn_results['rewards'],
        color="#D55E00", label='DQN')
ax.fill_between(ddqn_results['episodes'],
                ddqn_results['rewards'] + 2*np.sqrt(ddqn_results['variances']),
                ddqn_results['rewards'] - 2*np.sqrt(ddqn_results['variances']),
                facecolor="#D55E00", alpha=0.2)
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_xlim([0, 500])
ax.set_title(env_id)
plt.tight_layout()
plt.show()
