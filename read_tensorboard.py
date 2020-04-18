# %%
import os
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# %%
tb_path = 'C:\\repository\\drl\\tf_log'
tb_name = 'dqn_CartPole-v0_2020-04-17_11-08-25'
ea = EventAccumulator(os.path.join(tb_path, tb_name)).Reload()

tags = ea.Tags()['scalars']


tag = 'Episode_Reward'
tag_values = []
steps = []

for event in ea.Scalars(tag):
    tag_values.append(event.value)
    steps.append(event.step)

df = pd.DataFrame(data=np.array([steps, tag_values]).transpose(),
                  columns=['steps', 'value'])

csv_path = 'C:\\repository\\drl\\results\\logs\\'
csv_file_name = tb_name + '_' + tag + '.csv'
a = csv_path + csv_file_name
df.to_csv(csv_path + csv_file_name, index=False)
