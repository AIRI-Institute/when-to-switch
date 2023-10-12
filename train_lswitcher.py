import argparse
import os
import random
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method
from pathlib import Path
from typing import Union

import gym
import numpy as np
import torch
import wandb
from numpy import float32
from pydantic import BaseModel
from sample_factory.utils.utils import log
from torch.utils.data import DataLoader, Dataset

from agents.epom import EpomConfig, EPOM
from agents.replan import RePlan, RePlanConfig
from learning.epom_config import Environment
from policy_estimation.model import PolicyEstimationModel
from pomapf_env.pomapf_config import POMAPFConfig
from pomapf_env.wrappers import MatrixObservationWrapper
from train_epom import make_env


class EstimatorSettings(BaseModel):
    algo: Union[RePlanConfig, EpomConfig] = RePlanConfig()
    device: str = 'cuda'
    num_trials: int = 7
    num_process: int = 3
    num_epochs_for_trial: int = 3
    num_generator_chunks: int = 10
    generate_num_samples: int = 1000000
    experience_skip_percent: float = 0.8
    gamma: float = 0.99
    batch_size: int = 512
    data_loader_num_workers: int = 1
    shuffle: bool = True
    lr: float = 0.0001
    lr_schedule: list = [4, 5]
    train_dir: str = 'train_dir/estimator'
    collect_num_frames_for_chunk: int = None

    env_cfg: Environment = Environment(grid_config=POMAPFConfig(
        map_name='(wc3-[A-P]|sc1-[A-S]|sc1-TaleofTwoCities|street-[A-P]|mazes-s[0-9]_|mazes-s[1-3][0-9]_|random-s[0-9]_|random-s[1-3][0-9]_)',
        max_episode_steps=512))

    use_wandb: bool = True


class ExperienceHolder:
    def __init__(self, size):
        self.observations = []
        self.returns = []
        self.size = size

    def get_progress_str(self):
        return f"{len(self.observations)}/{self.size}"

    def __len__(self):
        return len(self.observations)

    def is_full(self):
        return self.size <= len(self.observations)

    def get_data(self):
        return self.observations, self.returns

    def store(self, observation, target_return):
        self.observations.append(observation)
        self.returns.append(target_return)


class PolicyEstimationWrapper(gym.Wrapper, ):
    def __init__(self, env, memory, cfg):
        super().__init__(env)

        self.cfg: EstimatorSettings = cfg
        self.memory = memory
        self._rewards = None
        self._observations = None
        self._dones = None

    def step(self, actions):
        obs, reward, done, infos = self.env.step(actions)
        for agent_idx in range(self.env.grid_config.num_agents):
            if not infos[agent_idx].get('is_active', True):
                continue

            self._rewards[agent_idx].append(reward[agent_idx])
            self._observations[agent_idx].append(obs[agent_idx])
            self._dones[agent_idx].append(done[agent_idx])
        return obs, reward, done, infos

    def get_discounted_return(self, rewards):
        result = []
        running = 0
        for reward in reversed(rewards):
            running = reward + running * self.cfg.gamma
            result.append(running)
        return list(reversed(result))

    def reset(self, **kwargs):
        if self._rewards:
            for agent_idx in range(self.env.grid_config.num_agents):
                returns = self.get_discounted_return(self._rewards[agent_idx])
                for step_idx, target_return in enumerate(returns):
                    if np.random.rand() >= self.cfg.experience_skip_percent:
                        self.memory.store(self._observations[agent_idx][step_idx], target_return)
        obs = self.env.reset(**kwargs)
        self._rewards = [[] for _ in range(self.env.grid_config.num_agents)]
        self._observations = [[] for _ in range(self.env.grid_config.num_agents)]
        self._dones = [[] for _ in range(self.env.grid_config.num_agents)]

        return obs


def collect_samples(cfg: EstimatorSettings):
    if cfg.algo.name == 'EPOM':
        algo = EPOM(cfg.algo)
    elif cfg.algo.name == 'RePlan':
        algo = RePlan(cfg.algo)
    else:
        raise KeyError(f"No algorithm with name: {cfg.algo}")

    experience_holder = ExperienceHolder(cfg.collect_num_frames_for_chunk)

    while not experience_holder.is_full():
        # Pick random number of agents
        cfg.env_cfg.grid_config.num_agents = random.choice([50, 100, 150, 200, 250, 300])
        env = make_env(cfg.env_cfg)
        env = PolicyEstimationWrapper(env, experience_holder, cfg)

        dones = [False]
        obs = env.reset()
        algo.after_reset()
        log.debug(
            f'collected: {experience_holder.get_progress_str()}, map: {env.grid_config.map_name}, num_agents: {env.grid_config.num_agents}')
        while not all(dones):
            action = algo.act(obs)
            obs, _, dones, info = env.step(action)
            algo.after_step(dones)
        env.reset()
    return experience_holder.get_data()


def generate_data_new(cfg: EstimatorSettings = EstimatorSettings()):
    observations = []
    returns = []

    with ProcessPoolExecutor(cfg.num_process) as executor:
        future_to_stuff = []
        # todo compute it automatically on the creation
        cfg.collect_num_frames_for_chunk = cfg.generate_num_samples // cfg.num_generator_chunks
        for _ in range(cfg.num_generator_chunks):
            future_to_stuff.append(executor.submit(collect_samples, cfg))
        for idx, future in enumerate(future_to_stuff):
            p_obs, p_returns = future.result()
            observations += p_obs
            returns += p_returns
            log.debug(f'Done {idx + 1} / {cfg.num_generator_chunks} of generator chunks')
            log.debug(f'Already collected {len(returns)}')
        log.debug(f'Done with collecting experience!')

    return observations, returns


class CustomDataset(Dataset):
    def __init__(self, observations, returns):
        self.observations = MatrixObservationWrapper.to_matrix(observations)
        self.returns = returns

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observations = self.observations[idx]
        returns = self.returns[idx]
        sample = {"features": observations, "targets": float32(returns)}
        return sample


def train(train_dataset, test_dataset, cfg, model):
    train_dataset = CustomDataset(*train_dataset)
    test_dataset = CustomDataset(*test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.data_loader_num_workers,
                              shuffle=cfg.shuffle)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.data_loader_num_workers,
                             shuffle=cfg.shuffle)
    loaders = OrderedDict(train=train_loader, valid=test_loader)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.lr_schedule:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_schedule)

    from catalyst.runners import SupervisedRunner
    runner = SupervisedRunner()

    runner.train(
        model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, loaders=loaders,
        logdir=cfg.train_dir, valid_loader="valid", valid_metric="loss", minimize_valid_metric=True,
        num_epochs=cfg.num_epochs_for_trial, verbose=False,
    )


def estimation_main_loop(cfg: EstimatorSettings):
    os.environ['OMP_NUM_THREADS'] = str(1)
    os.environ['MKL_NUM_THREADS'] = str(1)

    log.debug('Collecting frames for test dataset:')
    if cfg.use_wandb:
        wandb.tensorboard.patch(root_logdir=cfg.train_dir)
        wandb.init(project='LSwitcher', save_code=False, sync_tensorboard=True, anonymous="allow",
                   tags=[cfg.algo.name], )

    test_dataset = generate_data_new(cfg)

    # env = make_env(cfg.env_cfg)
    model = PolicyEstimationModel()

    for epoch in range(cfg.num_trials):
        log.debug('Collecting frames for train dataset:')
        train_dataset = generate_data_new(cfg)
        log.debug('Starting training process:')
        train(train_dataset=train_dataset, test_dataset=test_dataset, cfg=cfg, model=model)

        best_model_path = Path(cfg.train_dir) / 'checkpoints' / 'best.pth'
        log.debug(f'Loading best model from: {best_model_path}')

    if cfg.use_wandb:
        log.debug('Saving best model to wandb:')
        wandb.save(str(Path(cfg.train_dir) / 'checkpoints' / 'best.pth'))
        wandb.save(str(Path(cfg.train_dir) / 'checkpoints' / 'last.pth'))


def train_epom():
    set_start_method('spawn')
    es = EstimatorSettings(num_process=3, num_generator_chunks=12, algo=EpomConfig(path_to_weights='weights/epom'))
    estimation_main_loop(es)


def train_replan():
    es = EstimatorSettings(num_process=5, num_generator_chunks=80, algo=RePlanConfig())
    estimation_main_loop(es)


def main():
    parser = argparse.ArgumentParser(description='Policy estimation code')
    parser.add_argument('--algo', type=str, action="store", default='RePlan', help='RePlan/EPOM', required=False)
    args = parser.parse_args()

    if args.algo == 'RePlan':
        train_replan()
    elif args.algo == 'EPOM':
        train_epom()
    else:
        raise KeyError(f"No algo with name {args.algo}")


if __name__ == '__main__':
    main()
