def off_policy_trainer(env, agent, replay_buffer, writer, config):
    episode_rewards = []

    # Get samples from environment to warm up
    warm_up_step = 0
    while warm_up_step < config['warm_up_steps']:
        state = env.reset()
        while True:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            warm_up_step = warm_up_step + 1
            replay_buffer.push(state, action, reward, next_state, done)
            if done:
                break
            state = next_state

    # start training
    for episode in range(config['max_episodes']):
        state = env.reset()
        episode_reward = 0

        for step in range(config['max_steps']):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(replay_buffer) > config['batch_size']:
                batch_data = replay_buffer.sample(config['batch_size'])
                results_dict = agent.learn(batch_data)

                writer.add_scalar('Q_Net_Loss', results_dict['q_net_loss'], agent.steps_done)
                writer.add_scalar('Epsilon', results_dict['eps'], agent.steps_done)

            if done or step == config['max_steps'] - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

        writer.add_scalar('Episode Reward', episode_reward, episode + 1)

    return episode_rewards