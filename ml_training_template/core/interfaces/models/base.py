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


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

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
    print(dir(model))
    print(model)
