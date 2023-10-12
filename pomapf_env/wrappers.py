import re
from copy import deepcopy

import gym
import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box
from numpy import float32
from pogema import GridConfig

from pomapf_env.custom_maps import MAPS_REGISTRY


class RewardShaping(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._previous_xy = None

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        for agent_idx in range(self.get_num_agents()):
            reward = rewards[agent_idx]
            reward -= 0.0001
            if action[agent_idx] != 0:
                if tuple(self._previous_xy[agent_idx]) == tuple(observations[agent_idx]['xy']):
                    reward -= 0.0002
            rewards[agent_idx] = reward
            self._previous_xy[agent_idx] = observations[agent_idx]['xy']

        return observations, rewards, dones, infos

    def reset(self):
        observation = self.env.reset()
        self._previous_xy = [[0, 0] for _ in range(self.get_num_agents())]

        return observation


class MultiMapWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.grid_config.seed)
        pattern = self.grid_config.map_name

        if pattern:
            for map_name in MAPS_REGISTRY:
                if re.match(pattern, map_name):
                    cfg = deepcopy(self.grid_config)
                    cfg.map = MAPS_REGISTRY[map_name]
                    cfg.map_name = map_name
                    cfg = GridConfig(**cfg.dict())
                    self._configs.append(cfg)
            if not self._configs:
                raise KeyError(f"No map matching: {pattern}")

    def step(self, action):
        observations, rewards, done, info = self.env.step(action)
        cfg = self.grid_config
        if cfg.map_name:
            for agent_idx in range(self.get_num_agents()):
                if 'episode_extra_stats' in info[agent_idx]:
                    for key, value in list(info[agent_idx]['episode_extra_stats'].items()):
                        if key == 'Done':
                            continue
                        info[agent_idx]['episode_extra_stats'][f'{key}-{cfg.map_name.split("-")[0]}'] = value
        return observations, rewards, done, info

    def reset(self, **kwargs):
        if self._configs is not None and len(self._configs) >= 1:
            cfg = deepcopy(self._configs[self._rnd.integers(0, len(self._configs))])
            self.env.unwrapped.grid_config = cfg
        return self.env.reset(**kwargs)


class MatrixObservationWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        # full_size = self.config.obs_radius * 2 + 1
        full_size = self.env.observation_space['obstacles'].shape[0]
        self.observation_space = gym.spaces.Dict(
            obs=gym.spaces.Box(0.0, 1.0, shape=(3, full_size, full_size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
        )

    @staticmethod
    def get_square_target(x, y, tx, ty, obs_radius):
        full_size = obs_radius * 2 + 1
        result = np.zeros((full_size, full_size))
        dx, dy = x - tx, y - ty

        dx = min(dx, obs_radius) if dx >= 0 else max(dx, -obs_radius)
        dy = min(dy, obs_radius) if dy >= 0 else max(dy, -obs_radius)
        result[obs_radius - dx, obs_radius - dy] = 1
        return result

    @staticmethod
    def to_matrix(observations):
        result = []
        obs_radius = observations[0]['obstacles'].shape[0] // 2
        for agent_idx, obs in enumerate(observations):
            result.append(
                {"obs": np.concatenate([obs['obstacles'][None], obs['agents'][None],
                                        MatrixObservationWrapper.get_square_target(*obs['xy'], *obs['target_xy'],
                                                                                   obs_radius)[None]]).astype(float32),
                 "xy": np.array(obs['xy'], dtype=float32),
                 "target_xy": np.array(obs['target_xy'], dtype=float32),
                 })
        return result

    def observation(self, observation):
        result = self.to_matrix(observation)
        return result
