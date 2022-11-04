from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def metric(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Metrics to measure model performance"""
        raise NotImplementedError()
