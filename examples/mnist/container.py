from typing import Any, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from ml_training_template.core.interfaces import (
    BaseModel,
    BaseModelContainer,
    BaseOptimizer,
    BaseScheduler,
)
from ml_training_template.core.patterns.registry import ModelContainerRegistry


@ModelContainerRegistry.register()
class MNISTModelContainer(BaseModelContainer):
    """Abstract Class for Model Container"""

    def __init__(self,
                 model: Type["BaseModel"],
                 optimizer: Type["BaseOptimizer"],
                 scheduler: Optional["BaseScheduler"],
                 *args: Any, **kwargs: Any):
        """
        Args:
            model (nn.Module): PyTorch model
        """
        super().__init__(model=model, optimizer=optimizer,
                         scheduler=scheduler, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def shared_step(self, x: torch.Tensor, y: torch.Tensor):
        output = self.forward(x)
        loss = self.model.loss(output=output, target=y)
        return output, loss

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int):
        x, y = batch
        _, loss = self.shared_step(x=x, y=y)
        return {"train/loss": loss, "loss": loss}

    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output, loss = self.shared_step(x=x, y=y)
        metric = self.model.metric(output=output, target=y)
        return {
            "valid/loss": loss,
            "valid/metric": metric,
            "loss": loss,
        }

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc

    def test_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int):
        x, y = batch
        _, loss = self.shared_step(x=x, y=y)
        return {"test/loss": loss, "loss": loss}
