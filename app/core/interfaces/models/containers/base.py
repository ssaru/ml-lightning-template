from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
import torch.nn as nn


class BaseModelContainer(ABC):
    """Abstract Class for Model Container"""

    def __init__(self, model: nn.Module):
        """
        Args:
            model (nn.Module): PyTorch model
        """
        super().__init__()
        self.model = model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""
        Same as :meth:`torch.nn.Module.forward`.
        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.
        Return:
            Your model's output
        """
        raise self.model.forward(*args, **kwargs)

    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError()

    def shared_step(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int):
        raise NotImplementedError()

    def training_epoch_end(self, training_step_outputs):
        raise NotImplementedError()

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_epoch_end(self, validation_step_outputs):
        raise NotImplementedError()
