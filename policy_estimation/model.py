from argparse import Namespace

import gym
import torch
from gym.spaces import Box
from sample_factory.algorithms.appo.model_utils import nonlinearity
from sample_factory.utils.timing import Timing
from torch import nn

from learning.encoder import ResnetEncoder
from learning.epom_config import ExperimentSettings


class PolicyEstimationModel(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            exp_set = ExperimentSettings()
            cfg = {'full_config': {'experiment_settings': exp_set.dict()}, **exp_set.dict()}
            cfg = Namespace(**cfg)

        full_size = 5 * 2 + 1
        observation_space = gym.spaces.Dict(
            obs=gym.spaces.Box(0.0, 1.0, shape=(3, full_size, full_size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
        )

        self.encoder = ResnetEncoder(cfg, observation_space, Timing())
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.get_encoder_out_size(), 512),
            nonlinearity(cfg),
            nn.Linear(512, 512),
            nonlinearity(cfg),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.value_head(x)
        x = torch.squeeze(x, 1)
        return x
