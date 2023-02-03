from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseModel(ABC, nn.Module):

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def loss(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def metric(self, *args: Any, **kwargs: Any) -> Any:
        """Metrics to measure model performance"""
        raise NotImplementedError()
