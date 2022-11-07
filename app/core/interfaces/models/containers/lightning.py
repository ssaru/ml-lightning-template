from typing import Any, Tuple, Type

import torch
from pytorch_lightning import LightningModule

from app.core.interfaces.models.base import BaseModel
from app.core.interfaces.models.containers.base import BaseModelContainer


class LightningModelContainer(BaseModelContainer, LightningModule):
    def __init__(self, model: Type["BaseModel"], *args: Any, **kwargs: Any):
        super().__init__(model=model, *args, **kwargs)

    def configure_optimizers(self):
        raise NotImplementedError()

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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from app.core.interfaces.models.base import BaseModel

    class MnistModel(BaseModel):
        def __init__(self, num_classes=10):
            super(MnistModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

        def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.nll_loss(output, target)

        def metric(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                pred = torch.argmax(output, dim=1)
                correct = torch.sum(pred == target).item()
            return correct / len(target)

    model = MnistModel()
    model_container = LightningModelContainer(model=model)
    print(dir(model_container))
