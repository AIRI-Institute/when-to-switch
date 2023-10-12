import numpy as np
import cppimport.import_hook
from planning.planner import planner
from pogema import GridConfig

from planning.astar_no_grid import AStar, INF


class RePlanBase:
    def __init__(self, use_best_move: bool = True, max_steps: int = INF, algo_source='c++', seed=None,
                 ignore_other_agents=False):

        self.use_best_move = use_best_move
        gc: GridConfig = GridConfig()

        self.actions = {tuple(gc.MOVES[i]): i for i in range(len(gc.MOVES))}
        self.steps = 0

        if algo_source == 'c++':
            self.algo_source = planner
        else:
            self.algo_source = AStar
        self.planner = None
        self.max_steps = max_steps
        self.previous_positions = None
        self.rnd = np.random.default_rng(seed)
        self.ignore_other_agents = ignore_other_agents

    def act(self, obs, skip_agents=None):
        num_agents = len(obs)
        if self.planner is None:
            self.planner = [self.algo_source(self.max_steps) for _ in range(num_agents)]
        if self.previous_positions is None:
            self.previous_positions = [[] for _ in range(num_agents)]
        obs_radius = len(obs[0]['obstacles']) // 2
        action = []

        for k in range(num_agents):
            self.previous_positions[k].append(obs[k]['xy'])
            if obs[k]['xy'] == obs[k]['target_xy']:
                action.append(None)
                continue
            obstacles = np.transpose(np.nonzero(obs[k]['obstacles']))
            if self.ignore_other_agents:
                other_agents = None
            else:
                other_agents = np.transpose(np.nonzero(obs[k]['agents']))
            self.planner[k].update_obstacles(obstacles, other_agents,
                                             (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius))

            if skip_agents and skip_agents[k]:
                action.append(None)
                continue

            self.planner[k].update_path(obs[k]['xy'], obs[k]['target_xy'])
            path = self.planner[k].get_next_node(self.use_best_move)
            if path is not None and path[1][0] < INF:
                action.append(self.actions[(path[1][0] - path[0][0], path[1][1] - path[0][1])])
            else:
                action.append(None)
        self.steps += 1
        return action

    def get_path(self):
        results = []
        for idx in range(len(self.planner)):
            results.append(self.planner[idx].get_path(use_best_node=False))
        return results


class FixNonesWrapper:

    def __init__(self, agent):
        self.agent = agent
        self.rnd = self.agent.rnd
        # self.env = agent.env

    def act(self, obs, skip_agents=None):
        actions = self.agent.act(obs, skip_agents=skip_agents)
        for idx in range(len(actions)):
            if actions[idx] is None:
                actions[idx] = 0
        return actions


class NoPathSoRandomOrStayWrapper:

    def __init__(self, agent):
        self.agent = agent
        self.rnd = self.agent.rnd

    def act(self, obs, skip_agents=None):
        actions = self.agent.act(obs, skip_agents=skip_agents)
        for idx in range(len(actions)):
            if actions[idx] is None:
                if self.rnd.random() <= 0.5:
                    actions[idx] = 0
                else:
                    actions[idx] = self.get_random_move(obs, idx)
        return actions

    def get_random_move(self, obs, agent_id):
        deltas = GridConfig().MOVES
        actions = [1, 2, 3, 4]

        self.agent.rnd.shuffle(actions)
        for idx in actions:
            i = len(obs[agent_id]['obstacles']) // 2 + deltas[idx][0]
            j = len(obs[agent_id]['obstacles']) // 2 + deltas[idx][1]
            if obs[agent_id]['obstacles'][i][j] == 0:
                return idx
        return 0


class FixLoopsWrapper(NoPathSoRandomOrStayWrapper):
    def __init__(self, agent, stay_if_loop_prob=None, add_none_if_loop=False):
        super().__init__(agent)
        self.rnd = self.agent.rnd
        self.previous_positions = None
        self.stay_if_loop_prob = stay_if_loop_prob if stay_if_loop_prob else 0.5
        self.add_none_if_loop = add_none_if_loop

    def act(self, obs, skip_agents=None):
        num_agents = len(obs)
        if self.previous_positions is None:
            self.previous_positions = [[] for _ in range(num_agents)]

        actions = self.agent.act(obs, skip_agents=skip_agents)
        for idx in range(len(actions)):
            if actions[idx] is None:
                continue
            path = self.previous_positions[idx]
            if len(path) > 1:
                next_step = obs[idx]['xy']
                dx, dy = GridConfig().MOVES[actions[idx]]
                next_pos = dx + next_step[0], dy + next_step[1]
                if path[-1] == next_pos or path[-2] == next_pos:
                    if self.add_none_if_loop:
                        actions[idx] = None
                    elif next_pos == next_step:
                        actions[idx] = self.get_random_move(obs, idx)
                    elif self.rnd.random() < self.stay_if_loop_prob:
                        actions[idx] = 0
            self.previous_positions[idx].append(obs[idx]['xy'])
        return actions
