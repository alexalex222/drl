from tianshou.env.atari import create_atari_environment

env = create_atari_environment(
        'Pong', sticky_actions=False, max_episode_steps=2000)
state = env.reset()
action = env.action_space().sample()
next_state, reward, done, info = env.step(action)