import os

from prettytable import PrettyTable

from agents.assistant_switcher import example_assistant_switcher
from agents.epom import example_epom
from agents.heuristic_switcher import example_heuristic_switcher
from agents.learnable_switcher import example_learnable_switcher
from agents.replan import example_replan


def example(map_name='sc1-AcrosstheCape', num_agents=64, seed=0, animate=True):
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = "1"

    run_examples_funcs = [
        example_epom,
        example_replan,
        example_assistant_switcher,
        example_heuristic_switcher,
        example_learnable_switcher
    ]

    score_table = PrettyTable()
    score_table.field_names = ["Algorithm", "ISR", "CSR", "Episode Length"]

    for run_example_func in run_examples_funcs:
        result = run_example_func(map_name=map_name, num_agents=num_agents, seed=seed, animate=animate)

        if result:
            score_table.add_row([result['algorithm'], result['ISR'], result['CSR'], result['ep_length']])

            print(score_table.get_string(start=len(score_table._rows) - 1, end=len(score_table._rows)))

    print(score_table)


if __name__ == '__main__':
    example()
