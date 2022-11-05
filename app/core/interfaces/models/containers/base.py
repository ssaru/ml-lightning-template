from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class BaseModelContainer(LightningModule, ABC):
    """Abstract Class for Model Container"""

    def __init__(self, model: nn.Module):
        """
        Args:
            model (nn.Module): PyTorch model
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError()

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
