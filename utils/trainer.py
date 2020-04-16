

def off_policy_trainer(env,
                       agent,
                       replay_buffer,
                       writer,
                       config,
                       state_normalizer=None,
                       reward_normalizer=None):
    # Get samples from environment to warm up
    warm_up_step = 0
    while warm_up_step < config['warm_up_steps']:
        state = env.reset()
        while True:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            warm_up_step = warm_up_step + 1
            # normalized states if necessary
            if state_normalizer:
                state_normalized = state_normalizer(state)
                next_state_normalized = state_normalizer(next_state)
            else:
                state_normalized = state
                next_state_normalized = next_state
            # normalized reward if necessary
            if reward_normalizer:
                reward_normalized = reward_normalizer(reward)
            else:
                reward_normalized = reward

            replay_buffer.push(state_normalized, action, reward_normalized, next_state_normalized, done)
            if done:
                break
            state = next_state

    # start training
    episode_rewards = []
    for episode in range(config['max_episodes']):
        state = env.reset()
        episode_reward = 0

        for step in range(config['max_steps']):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # normalized states if necessary
            if state_normalizer:
                state_normalized = state_normalizer(state)
                next_state_normalized = state_normalizer(next_state)
            else:
                state_normalized = state
                next_state_normalized = next_state
            # normalized reward if necessary
            if reward_normalizer:
                reward_normalized = reward_normalizer(reward)
            else:
                reward_normalized = reward

            replay_buffer.push(state_normalized, action, reward_normalized, next_state_normalized, done)

            if len(replay_buffer) > config['batch_size']:
                batch_data = replay_buffer.sample(config['batch_size'])
                results_dict = agent.learn(batch_data)

                if results_dict:
                    writer.add_scalar('Q_Net_Loss', results_dict['q_net_loss'], agent.steps_done)
                    writer.add_scalar('Epsilon', results_dict['eps'], agent.steps_done)

            if done or step == config['max_steps'] - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

        writer.add_scalar('Episode Reward', episode_reward, episode + 1)

    return episode_rewards
