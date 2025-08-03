import gymnasium as gym
import numpy as np
import pandas as pd


class CryptoTradingEnv(gym.Env):
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path)
        self.current_step = 0
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        return (
            self._get_observation(),
            self._get_reward(action),
            self.current_step >= len(self.data) - 1,
            False,
            {},
        )

    def _get_observation(self):
        return self.data.iloc[self.current_step].values

    def _get_reward(self, action):
        return 0

    def _take_action(self, action):
        return action
