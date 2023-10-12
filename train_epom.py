import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import wandb
import yaml
from sample_factory.algorithms.utils.algo_utils import EXTRA_EPISODIC_STATS_PROCESSING, EXTRA_PER_POLICY_SUMMARIES
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm
from sample_factory.utils.utils import log

# noinspection PyUnresolvedReferences
from learning.encoder import ResnetEncoder
from learning.epom_config import Environment, Experiment
from learning.grid_memory import GridMemoryWrapper
from pomapf_env.env import make_pomapf
from pomapf_env.wrappers import MatrixObservationWrapper


def make_env(env_cfg: Environment = Environment()):
    env = make_pomapf(grid_config=env_cfg.grid_config)
    return env


def create_pogema_env(full_env_name, cfg=None, env_config=None):
    environment_config: Environment = Environment(**cfg.full_config['environment'])
    env = make_env(environment_config)
    gm_radius = environment_config.grid_memory_obs_radius
    env = GridMemoryWrapper(env, obs_radius=gm_radius if gm_radius else environment_config.grid_config.obs_radius)
    env = MatrixObservationWrapper(env)
    return env


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='POMAPF',
        make_env_func=create_pogema_env,
    )

    EXTRA_EPISODIC_STATS_PROCESSING.append(pogema_extra_episodic_stats_processing)
    EXTRA_PER_POLICY_SUMMARIES.append(pogema_extra_summaries)


def pogema_extra_episodic_stats_processing(policy_id, stat_key, stat_value, cfg):
    pass


def pogema_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    for key in policy_avg_stats:
        if key in ['reward', 'len', 'true_reward', 'Done']:
            continue

        avg = np.mean(np.array(policy_avg_stats[key][policy_id]))
        summary_writer.add_scalar(key, avg, env_steps)
        log.debug(f'{policy_id}-{key}: {round(float(avg), 3)}')


def validate_config(config):
    exp = Experiment(**config)
    flat_config = Namespace(**exp.async_ppo.dict(),
                            **exp.experiment_settings.dict(),
                            **exp.global_settings.dict(),
                            **exp.evaluation.dict(),
                            full_config=exp.dict()
                            )
    return exp, flat_config


def select_free_dir_name(rood_dir, max_id=100000):
    for cnt in range(1, max_id):
        free_folder = f"{cnt}".zfill(4)
        full_path = Path(rood_dir) / Path(free_folder)
        if not full_path.exists():
            return free_folder
    raise KeyError(f"Can't select a folder in {max_id} attempts")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process training config.')

    parser.add_argument('--config_path', type=str, action="store", default='configs/train-debug.yaml',
                        help='path to yaml file with single run configuration', required=False)

    parser.add_argument('--wandb_thread_mode', type=bool, action='store', default=False,
                        help='Run wandb in thread mode. Usefull for some setups.', required=False)

    params = parser.parse_args()

    register_custom_components()
    if params.config_path is None:
        raise ValueError("You should specify --config_path or --raw_config argument!")
    with open(params.config_path, "r") as f:
        config = yaml.safe_load(f)

    exp, flat_config = validate_config(config)
    if exp.global_settings.experiments_root is None:
        exp.global_settings.experiments_root = select_free_dir_name(exp.global_settings.train_dir)
        exp, flat_config = validate_config(exp.dict())
    log.debug(exp.global_settings.experiments_root)

    if exp.global_settings.use_wandb:
        import os
        if params.wandb_thread_mode:
            os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project=exp.name, config=exp.dict(), save_code=False, sync_tensorboard=True, anonymous="allow", )

    status = run_algorithm(flat_config)

    return status


if __name__ == '__main__':
    sys.exit(main())
