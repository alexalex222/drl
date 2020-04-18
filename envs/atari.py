import random
import numpy as np
import gym
import cv2


class AtariGame:
    def __init__(self, env_name, no_op_steps=30):
        self.env = gym.make(env_name)
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = 4
        self._width = 84
        self._height = 84
        self.stacked_frames = np.zeros((self.agent_history_length, self._width, self._height), dtype=np.uint8)

    def reset(self):
        frame = self.env.reset()

        if 'FIRE' in self.env.unwrapped.get_action_meanings():
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            for _ in range(random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1)  # Action 'Fire'
        self.last_lives = self.env.unwrapped.ale.lives()
        processed_frame = self.process_frame(frame)
        self.stacked_frames = np.repeat(processed_frame, self.agent_history_length, axis=0)

        return self.stacked_frames

    def step(self, action):
        new_frame, reward, terminal, info = self.env.step(action)
        if self.env.unwrapped.ale.lives() < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = self.env.unwrapped.ale.lives()
        processed_new_frame = self.process_frame(new_frame)
        self.stacked_frames = np.append(self.stacked_frames[1:, :, :], processed_new_frame, axis=0)
        return self.stacked_frames, reward, terminal_life_lost, info

    def process_frame(self, frame):
        # convert to grey scale
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # resize 84 x 84
        transformed_frame = cv2.resize(
            grey_frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        # convert to numpy
        np_frame = np.asarray(transformed_frame, dtype=np.uint8)
        # reshape to 84 x 84 x 1
        np_frame = np.expand_dims(np_frame, 0)
        return np_frame

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(self._width, self._height, 4), dtype=np.uint8)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def metadata(self):
        return self.env.metadata

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps

    def close(self):
        return self.env.close()
