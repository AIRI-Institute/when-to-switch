import gym
import numpy as np
from gym.spaces import Box
from pogema import GridConfig
from pogema.grid import Grid


class GridMemory:
    def __init__(self, start_r=32):
        self.memory = np.zeros(shape=(start_r * 2 + 1, start_r * 2 + 1))

    @staticmethod
    def try_to_insert(x, y, source, target):
        r = source.shape[0] // 2
        try:
            target[x - r:x + r + 1, y - r:y + r + 1] = source
            return True
        except ValueError:
            return False

    def increase_memory(self):
        m = self.memory
        r = self.memory.shape[0]
        self.memory = np.zeros(shape=(r * 2 + 1, r * 2 + 1))
        assert self.try_to_insert(r, r, m, self.memory)

    def update(self, x, y, obstacles):
        while True:
            r = self.memory.shape[0] // 2
            if self.try_to_insert(r + x, r + y, obstacles, self.memory):
                break
            self.increase_memory()

    def get_observation(self, x, y, obs_radius):
        while True:
            r = self.memory.shape[0] // 2
            tx, ty = x + r, y + r
            size = self.memory.shape[0]
            if 0 <= tx - obs_radius and tx + obs_radius + 1 <= size:
                if 0 <= ty - obs_radius and ty + obs_radius + 1 <= size:
                    return self.memory[tx - obs_radius:tx + obs_radius + 1, ty - obs_radius:ty + obs_radius + 1]

            self.increase_memory()

    def render(self):
        m = self.memory.astype(int).tolist()
        gc = GridConfig(map=m)
        g = Grid(add_artificial_border=False, grid_config=gc)
        r = self.memory.shape[0] // 2
        g.positions_xy = [[r, r]]
        g.finishes_xy = []
        g.render()


class MultipleGridMemory:
    def __init__(self):
        self.memories = None

    def update(self, observations):
        if self.memories is None or len(self.memories) != len(observations):
            self.memories = [GridMemory() for _ in range(len(observations))]
        for agent_idx, obs in enumerate(observations):
            self.memories[agent_idx].update(*obs['xy'], obs['obstacles'])

    def get_observations(self, xy_list, obs_radius):
        return [self.memories[idx].get_observation(x, y, obs_radius) for idx, (x, y) in enumerate(xy_list)]

    def modify_observation(self, observations, obs_radius):
        all_xy = [observations[idx]['xy'] for idx in range(len(observations))]
        for obs, gm_obs in zip(observations, self.get_observations(all_xy, obs_radius)):
            obs['obstacles'] = gm_obs

        r = obs_radius
        rr = observations[0]['agents'].shape[0] // 2
        for agent_idx, obs in enumerate(observations):

            if rr <= r:
                agents = np.zeros(shape=(r * 2 + 1, r * 2 + 1))
                agents[r - rr:r + rr + 1, r - rr: r + rr + 1] = obs['agents']
                obs['agents'] = agents
            else:
                obs['agents'] = obs['agents'][rr - r:rr + r + 1, rr - r: rr + r + 1]

    def clear(self):
        self.memories = None


class GridMemoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_radius):
        super().__init__(env)
        self.obs_radius = obs_radius

        size = self.obs_radius * 2 + 1
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            obstacles=gym.spaces.Box(0.0, 1.0, shape=(size, size)),
            agents=gym.spaces.Box(0.0, 1.0, shape=(size, size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
        )

        self.mgm = MultipleGridMemory()

    def observation(self, observations):
        self.mgm.update(observations)
        self.mgm.modify_observation(observations, self.obs_radius)
        return observations

    def reset(self):
        self.mgm.clear()
        return self.observation(self.env.reset())
