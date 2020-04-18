from envs.atari import AtariGame


# %%
env_id = 'BreakoutNoFrameskip-v4'
env = AtariGame(env_id)
state = env.reset()

i = 0
while True:
    print(i)
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    if done:
        break
