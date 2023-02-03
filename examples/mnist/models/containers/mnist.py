from typing import Tuple

import torch
import torch.nn as nn

from ml_training_template.core.interfaces.models.containers import (
    BaseModelContainer,
)
from ml_training_template.core.patterns.registry import ModelContainerRegistry


@ModelContainerRegistry.register("MNIST")
class MNISTModelContainer(BaseModelContainer):
    """Abstract Class for Model Container"""

    def __init__(self, model: nn.Module):
        """
        Args:
            model (nn.Module): PyTorch model
        """
        super().__init__(model=model)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def configure_optimizers(self):
        opt_args = dict(self.config.optimizer.params)
        opt_args.update({"params": self.model.parameters(), "lr": self.lr})

        # NOTE. 아래 두 코드는 Abstract코드에 있는 것이 더 좋을 것 같음
        # NOTE. Optimizer를 Registry에서 가져오는 코드를 짜야함
        #opt = load_class(module=optim, name=self.config.optimizer.type, args=opt_args)

        # NOTE. Scheduler를 Registry에서 가져오는 코드를 짜야함
        # scheduler_args = dict(self.config.scheduler.params)
        # scheduler_args.update({"optimizer": opt, "gamma": self.scheduler_gamma})
        # scheduler = load_class(
        #     module=optim.lr_scheduler,
        #     name=self.config.scheduler.type,
        #     args=scheduler_args,
        # )

        # result = {"optimizer": opt, "lr_scheduler": scheduler}
        return None

    def shared_step(self, x: torch.Tensor, y: torch.Tensor):
        output = self.forward(x)
        loss = self.model.loss(output=output, target=y)
        return output, loss

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int):
        x, y = batch
        _, loss = self.shared_step(x=x, y=y)
        return {"train/loss": loss}

    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output, loss = self.shared_step(x=x, y=y)
        metric = self.model.metric(output=output, target=y)
        return {
            "valid/loss": loss,
            "valid/metric": metric,
        }

    def validation_epoch_end(self, validation_step_outputs):
        pass
