import time
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
import gym
from dqn.dqn import DQN
import utils


# Hyper Parameters
MaxEpisodes = 300
num_of_runs = 5
winWidth = 100
writeCSV = True
savePlot = True

config_file = 'ddqn_cartpole.json'


if __name__ == '__main__':
	aver_rwd_dqn = np.array((MaxEpisodes, ))
	config = json.load(open('dqn/configs/' + config_file))

	for exp in range(num_of_runs):
		print('\nExperiment NO.' + str(exp+1))

		dqn_agent = DQN(config)

		rwd_dqn = dqn_agent.train_agent(MaxEpisodes)

		del dqn_agent

		# incrementally calculate mean and variance
		rwd_dqn = utils.misc.moving_avg(rwd_dqn, winWidth)
		tmp_rwd = np.array(rwd_dqn)
		pre_rwd = aver_rwd_dqn
		aver_rwd_dqn = aver_rwd_dqn + (tmp_rwd - aver_rwd_dqn)/float(exp+1)
		if exp == 0:
			var_rwd = np.zeros(aver_rwd_dqn.shape)
		else:
			var_rwd = var_rwd + np.multiply((tmp_rwd - pre_rwd), (tmp_rwd - aver_rwd_dqn))/float(exp+1)
			var_rwd = var_rwd/float(exp+1)

	# save data to csv
	if writeCSV:
		data = {'episodes': range(MaxEpisodes), 'rewards': list(aver_rwd_dqn), 'variances': list(np.sqrt(var_rwd))}
		df = pd.DataFrame(data=dict([(key, pd.Series(value)) for key, value in data.items()]),
			index=range(0, MaxEpisodes),
			columns=['episodes', 'rewards', 'variances'])
		if config['double_q_model']:
			if config['memory']['prioritized']:
				df.to_csv('results/logs/ddqn_pri_steps_{}_run_{}.csv'.format(MaxEpisodes, num_of_runs))
			else:
				df.to_csv('results/logs/ddqn_steps_{}_run_{}.csv'.format(MaxEpisodes, num_of_runs))
		else:
			if config['memory']['prioritized']:
				df.to_csv('results/logs/dqn_pri_steps_{}_run_{}.csv'.format(MaxEpisodes, num_of_runs))
			else:
				df.to_csv('results/logs/dqn_steps_{}_run_{}.csv'.format(MaxEpisodes, num_of_runs))

	# Save reward plot
	fig, ax = plt.subplots(nrows=1, ncols=1)
	ax.plot(range(MaxEpisodes), list(aver_rwd_dqn), 'k', label='no pre-training')
	ax.fill_between(range(MaxEpisodes), aver_rwd_dqn + np.sqrt(var_rwd), aver_rwd_dqn - np.sqrt(var_rwd), facecolor='black', alpha=0.1)
	ax.set_title('Number of Run: ' + str(num_of_runs))
	ax.set_xlabel('Episodes')
	ax.set_ylabel('Average Rewards')
	ax.legend(loc='upper left')
	ax.grid()
	if config['double_q_model']:
		if config['memory']['prioritized']:
			fig.savefig('results/figures/ddqn_pri_{}.png'.format(int(time.time())))
		else:
			fig.savefig('results/figures/ddqn_{}.png'.format(int(time.time())))
	else:
		if config['memory']['prioritized']:
			fig.savefig('results/figures/dqn_pri_{}.png'.format(int(time.time())))
		else:
			fig.savefig('results/figures/dqn_{}.png'.format(int(time.time())))
	plt.close(fig)

