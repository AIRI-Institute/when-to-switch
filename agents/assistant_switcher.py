from pathlib import Path

from agents.utils_agents import run_algorithm

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import Extra

from agents.replan import RePlanConfig
from agents.utils_switching import SwitcherBaseConfig, SwitcherBase


class ASwitcherConfig(SwitcherBaseConfig, extra=Extra.forbid):
    name: Literal['ASwitcher'] = 'ASwitcher'
    planning: RePlanConfig = RePlanConfig(name='RePlanCPP', fix_loops=True, add_none_if_loop=True, no_path_random=False,
                                          use_best_move=False, fix_nones=False)


class AssistantSwitcher(SwitcherBase):

    def get_learning_use_mask(self, planning_actions, learning_actions, observations):
        return [a is None for a in planning_actions]


def example_assistant_switcher(map_name='sc1-AcrosstheCape', max_episode_steps=512, seed=None, num_agents=64,
                               main_dir='./', animate=False):
    from agents.epom import EpomConfig
    algo = AssistantSwitcher(ASwitcherConfig(learning=EpomConfig(path_to_weights=str(main_dir / Path('weights/epom')))))
    return run_algorithm(algo, map_name, max_episode_steps, seed, num_agents, animate)


if __name__ == '__main__':
    print(example_assistant_switcher(main_dir='../'))
