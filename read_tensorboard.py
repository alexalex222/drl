# %%
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# %%
task = 'MountainCar'
tb_path = 'C:\\temp\\tb_log\\*{0}*'.format(task)
all_rewards = []
for tb_file in glob.glob(tb_path):
    print(tb_file)
    ea = EventAccumulator(tb_file).Reload()

    tags = ea.Tags()['scalars']

    tag = 'Episode_Reward/eval'
    tag_values = []

    for event in ea.Scalars(tag):
        tag_values.append(event.value)

    all_rewards.append(tag_values)


all_rewards = np.asarray(all_rewards).transpose()
csv_path = 'C:\\repository\\drl\\results\\logs\\'
csv_file_name = task + '_temp' + '.csv'
np.savetxt(csv_path + csv_file_name, all_rewards, delimiter=',')


