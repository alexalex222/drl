import numpy as np
import gym


def off_policy_trainer(env,
                       agent,
                       replay_buffer,
                       writer,
                       config,
                       state_normalizer=None,
                       reward_normalizer=None):
    # Get samples from environment to warm up
    print('Warm up...')
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
    print('Start training')
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
                    if type(agent).__name__ == 'DQNAgent' or type(agent).__name__ == 'Categorical_DQNAgent':
                        writer.add_scalar('Q_Net_Loss', results_dict['q_net_loss'], agent.steps_done)
                        writer.add_scalar('Epsilon', results_dict['eps'], agent.steps_done)
                    elif type(agent).__name__ == 'SoftActorCriticAgent':
                        writer.add_scalar('Loss/value_net_loss', results_dict['value_loss'], agent.steps_done)
                        writer.add_scalar('Loss/q_net_loss1', results_dict['q1_loss'], agent.steps_done)
                        writer.add_scalar('Loss/q_net_loss2', results_dict['q2_loss'], agent.steps_done)
                        writer.add_scalar('Loss/policy_net_loss', results_dict['policy_loss'], agent.steps_done)
                    elif type(agent).__name__ == 'DeepDeterministicPolicyGradientAgent':
                        writer.add_scalar('Loss/policy_loss', results_dict['policy_loss'], agent.steps_done)
                        writer.add_scalar('Loss/critic_loss', results_dict['critic_loss'], agent.steps_done)

            if done:
                break

            state = next_state

        episode_rewards.append(episode_reward)
        print("Episode " + str(episode) + ": " + str(episode_reward))
        writer.add_scalar('Episode Reward/train', episode_reward, episode + 1)

        if (episode + 1) % config['eval_interval'] == 0:
            eval_episode_rewards = []
            for eval_ep in range(config['eval_episodes']):
                state = env.reset()
                eval_episode_reward = 0
                for _ in range(config['max_steps']):
                    action = agent.get_action_eval(state)
                    next_state, reward, done, _ = env.step(action)
                    eval_episode_reward += reward
                    if done:
                        break
                    state = next_state
                eval_episode_rewards.append(eval_episode_reward)
            average_eval_reward = np.mean(eval_episode_rewards)
            print("Eval: " + str(average_eval_reward))
            writer.add_scalar('Episode Reward/eval', average_eval_reward, episode + 1)

    return episode_rewards


def off_policy_trainer_gp(env,
                          agent,
                          replay_buffer,
                          writer,
                          config,
                          state_normalizer=None,
                          reward_normalizer=None):
    # Get samples from environment to warm up
    print('warm up...')
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
    print('Start training')
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

                inducing_batch = replay_buffer.sample(config['inducing_size'])
                agent.update_gp(inducing_batch)

                if results_dict:
                    writer.add_scalar('Q_Net_Loss', results_dict['q_net_loss'], agent.steps_done)
                    writer.add_scalar('Epsilon', 1 - agent.greedy_selection/agent.steps_done, agent.steps_done)

            if done:
                break

            state = next_state

        episode_rewards.append(episode_reward)
        print("Episode " + str(episode) + ": " + str(episode_reward))
        writer.add_scalar('Episode Reward', episode_reward, episode + 1)

        if (episode + 1) % config['eval_interval'] == 0:
            eval_episode_rewards = []
            for eval_ep in range(config['eval_episodes']):
                state = env.reset()
                eval_episode_reward = 0
                for _ in range(config['max_steps']):
                    action = agent.get_action_eval(state)
                    next_state, reward, done, _ = env.step(action)
                    eval_episode_reward += reward
                    if done:
                        break
                    state = next_state
                eval_episode_rewards.append(eval_episode_reward)
            average_eval_reward = np.mean(eval_episode_rewards)
            print("Eval: " + str(average_eval_reward))
            writer.add_scalar('Episode Reward/eval', average_eval_reward, episode + 1)

    return episode_rewards


def on_policy_trainer(env,
                       agent,
                       replay_buffer,
                       writer,
                       config,
                       state_normalizer=None,
                       reward_normalizer=None):
    # Get samples from environment to warm up
    print('Warm up...')
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
    print('Start training')
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

            if done:
                break

            state = next_state

        episode_rewards.append(episode_reward)
        print("Episode " + str(episode) + ": " + str(episode_reward))
        writer.add_scalar('Episode Reward/train', episode_reward, episode + 1)

        if (episode + 1) % config['learn_interval'] == 0:
            full_batch = replay_buffer.sample_sequence(batch_size=0)
            replay_buffer.reset()
            results_dict = agent.learn(full_batch)

            writer.add_scalar('Loss/total_loss', results_dict['total_loss'], agent.steps_done)
            writer.add_scalar('Loss/actor_loss', results_dict['policy_loss'], agent.steps_done)
            writer.add_scalar('Loss/value_loss', results_dict['value_loss'], agent.steps_done)
            writer.add_scalar('Loss/entropy_loss', results_dict['entropy_loss'], agent.steps_done)

        if (episode + 1) % config['eval_interval'] == 0:
            eval_episode_rewards = []
            for eval_ep in range(config['eval_episodes']):
                state = env.reset()
                eval_episode_reward = 0
                for _ in range(config['max_steps']):
                    action = agent.get_action_eval(state)
                    next_state, reward, done, _ = env.step(action)
                    eval_episode_reward += reward
                    if done:
                        break
                    state = next_state
                eval_episode_rewards.append(eval_episode_reward)
            average_eval_reward = np.mean(eval_episode_rewards)
            print("Eval: " + str(average_eval_reward))
            writer.add_scalar('Episode Reward/eval', average_eval_reward, episode + 1)

    return episode_rewards
