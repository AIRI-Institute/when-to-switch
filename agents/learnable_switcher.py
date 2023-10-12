from pathlib import Path

from pydantic import Extra

from agents.replan import RePlanConfig
from agents.utils_agents import run_algorithm
from agents.utils_switching import SwitcherBase, SwitcherBaseConfig
from policy_estimation.policy_estimator import PolicyEstimator
from pomapf_env.wrappers import MatrixObservationWrapper

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class LSwitcherConfig(SwitcherBaseConfig, extra=Extra.forbid):
    name: Literal['LSwitcher'] = 'LSwitcher'
    planning: RePlanConfig = RePlanConfig(fix_loops=True, add_none_if_loop=False, no_path_random=True,
                                          use_best_move=True, fix_nones=True)
    planning_path: str = "weights/pe-replan"
    learning_path: str = "weights/pe-epom"
    min_consequence_steps: int = 50


class LearnableSwitcher(SwitcherBase):
    def __init__(self, algo_cfg):
        super().__init__(algo_cfg)

        self.learning_estimator = PolicyEstimator()
        self.learning_estimator.load(algo_cfg.learning_path)

        self.planning_estimator = PolicyEstimator()
        self.planning_estimator.load(algo_cfg.planning_path)

        self._consequence_steps = None
        self._previous_mask = None

    def get_learning_use_mask(self, planning_actions, learning_actions, observations):
        num_agents = len(observations)
        if self._consequence_steps is None:
            self._consequence_steps = [0 for _ in range(num_agents)]
        learning_values = self.learning_estimator.predict(MatrixObservationWrapper.to_matrix(observations))
        planning_values = self.planning_estimator.predict(MatrixObservationWrapper.to_matrix(observations))

        mask = [learning_values[agent_idx] > planning_values[agent_idx] for agent_idx in range(num_agents)]
        if self._previous_mask is not None:
            for agent_idx in range(num_agents):
                if mask[agent_idx] != self._previous_mask[agent_idx]:
                    if self._consequence_steps[agent_idx] < self.cfg.min_consequence_steps:
                        mask[agent_idx] = self._previous_mask[agent_idx]
                    else:
                        self._consequence_steps[agent_idx] = 0
                self._consequence_steps[agent_idx] += 1
        self._previous_mask = mask

        return mask

    def after_reset(self):
        super().after_reset()
        self._consequence_steps = None
        self._previous_mask = None


def example_learnable_switcher(map_name='sc1-AcrosstheCape', max_episode_steps=512, seed=None, num_agents=64,
                               main_dir='./', animate=False):
    from agents.epom import EpomConfig
    algo = LearnableSwitcher(LSwitcherConfig(learning=EpomConfig(path_to_weights=str(main_dir / Path('weights/epom'))),
                                             planning_path=str(main_dir / Path('weights/pe-replan')),
                                             learning_path=str(main_dir / Path('weights/pe-epom'))))

    return run_algorithm(algo, map_name, max_episode_steps, seed, num_agents, animate)


if __name__ == '__main__':
    print(example_learnable_switcher(main_dir='../'))
