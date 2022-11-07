from abc import ABC
from typing import Any, Tuple, Type, Union

import torch
from pytorch_lightning import LightningModule

from app.core.interfaces.models import BaseModel
from app.core.interfaces.optimizer import BaseOptimizer
from app.core.interfaces.scheduler import BaseScheduler


class BaseModelContainer(ABC, LightningModule):
    """Abstract Class for Model Container"""

    def __init__(self,
                 model: Type["BaseModel"],
                 optimizer: Type["BaseOptimizer"],
                 scheduler: Union[None, Type["BaseScheduler"]],
                 *args: Any, **kwargs: Any):
        """
        Args:
            model (nn.Module): PyTorch model
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

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

    def configure_optimizers(self):
        if self.scheduler:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
        else:
            return {"optimizer": self.optimizer}

    def shared_step(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError()

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int):
        raise NotImplementedError()

    def training_epoch_end(self, training_step_outputs):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_epoch_end(self, validation_step_outputs):
        raise NotImplementedError()


if __name__ == "__main__":
    """PyTorch-Lightning Module과 BaseModelContainer의 property가 잘 있는지 확인"""
    model_container = BaseModelContainer(
        model=None, optimizer=None, scheduler=None)
    print(dir(model_container))
