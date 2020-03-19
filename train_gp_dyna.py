from datetime import datetime
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
import gym
from gp_dyna.gp_dyna import GpDynaQ
import utils

# Hyper Parameters
MaxEpisodes = 1000
num_of_runs = 5
winWidth = 100
writeCSV = True
savePlot = True

# config_file = 'gp_dyna_cartpole.json'
config_file = 'gp_dyna_acrobot.json'

if __name__ == '__main__':
    aver_rwd_dyna = np.array((MaxEpisodes,))
    config = json.load(open('gp_dyna/configs/' + config_file))

    for exp in range(num_of_runs):
        print('\nExperiment NO.' + str(exp + 1))

        dyna_q_agent = GpDynaQ(config)

        rwd_dyna = dyna_q_agent.train_agent(MaxEpisodes)

        del dyna_q_agent

        # incrementally calculate mean and variance
        rwd_dyna = utils.misc.moving_avg(rwd_dyna, winWidth)
        tmp_rwd = np.array(rwd_dyna)
        pre_rwd = aver_rwd_dyna
        aver_rwd_dyna = aver_rwd_dyna + (tmp_rwd - aver_rwd_dyna) / float(exp + 1)
        if exp == 0:
            var_rwd_dyna = np.zeros(aver_rwd_dyna.shape)
        else:
            var_rwd_dyna = var_rwd_dyna + np.multiply((tmp_rwd - pre_rwd), (tmp_rwd - aver_rwd_dyna)) / float(exp + 1)
            var_rwd_dyna = var_rwd_dyna / float(exp + 1)

    # save data to csv
    if writeCSV:
        data = {'episodes': range(MaxEpisodes), 'rewards': list(aver_rwd_dyna),
                'variances': list(np.sqrt(var_rwd_dyna))}
        df = pd.DataFrame(data=dict([(key, pd.Series(value)) for key, value in data.items()]),
                          index=range(0, MaxEpisodes),
                          columns=['episodes', 'rewards', 'variances'])
        df.to_csv('results/logs/gp_dyna_q_{}_run_{}.csv'.format(config['env_id'], datetime.now().strftime("%Y-%m-%d, %H-%M-%S")))

    # Save reward plot
    if savePlot:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(range(MaxEpisodes), list(aver_rwd_dyna), 'k', label='GP-Dyna-Q')
        ax.fill_between(range(MaxEpisodes), aver_rwd_dyna + np.sqrt(var_rwd_dyna),
                        aver_rwd_dyna - np.sqrt(var_rwd_dyna), facecolor='black', alpha=0.2)
        ax.set_title('Number of Run: ' + str(num_of_runs))
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Average Rewards')
        ax.legend(loc='upper left')
        ax.grid()
        fig.savefig('results/figures/gp_dyna_q_{}_{}.png'.format(config['env_id'], datetime.now().strftime("%Y-%m-%d, %H-%M-%S")))
        plt.close(fig)
