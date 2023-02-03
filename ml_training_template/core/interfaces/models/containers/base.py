from abc import ABC
from typing import Any, Dict, Optional, Tuple, Type

import torch
from pytorch_lightning import LightningModule

from ml_training_template.core.interfaces.models import BaseModel
from ml_training_template.core.interfaces.optimizer import BaseOptimizer
from ml_training_template.core.interfaces.scheduler import BaseScheduler


class BaseModelContainer(ABC, LightningModule):
    """Abstract Class for Model Container"""

    def __init__(self,
                 model: Type[BaseModel],
                 optimizer: Type[BaseOptimizer],
                 scheduler: Optional[BaseScheduler],
                 *args: Any, **kwargs: Any):
        """
        Args:
            model (nn.Module): PyTorch model
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

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
