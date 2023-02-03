# ML Lightning Template

Template code to easily create deep learning models

## Objective

If you create a model without worrying about training and storing the model, it is packaged to a level that can be learned and used immediately.

**[Configuration]**

```yaml
data:
    train:
        dataset:
            name: MNIST
            params:
                root: ./.data
                train: true
                download: true
                transform:
                    name: ToTensor
        dataloader:
            name: BaseDataLoader
            params:
                batch_size: 256
                shuffle: true
    valid:
        dataset:
            name: MNIST
            params:
                root: ./.data
                train: false
                download: true
                transform:
                    name: ToTensor
        dataloader:
            name: BaseDataLoader
            params:
                batch_size: 256
                shuffle: false
    test:
        dataset:
            name: MNIST
            params:
                root: .data
                train: false
                download: true
                transform:
                    name: ToTensor
        dataloader:
            name: BaseDataLoader
            params:
                batch_size: 256
                shuffle: false

container:
    name: MNISTModelContainer

    model:
        name: MnistModel
        params:
            num_classes: 10

    optimizer:
        name: SGD
        params:
            lr: 0.001
            momentum: 0.9

    scheduler:
        name: StepLR
        params:
            step_size: 30
            gamma: 0.1

trainer:
    name: BaseTrainer
    params:
        num_sanity_val_steps: 2
        enable_checkpointing: true
        max_epochs: 30

model_checkpoint:
    name: ModelCheckpoint
    params:
        dirpath: ./outputs
        filename: mnist
```

**[User Code]**

```python
from typing import Any, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional import accuracy

from ml_training_template.core.interfaces import (
    BaseModel,
    BaseModelContainer,
    BaseOptimizer,
    BaseScheduler,
)
from ml_training_template.core.patterns.registry import ModelRegistry
from ml_training_template.core.patterns.registry import ModelContainerRegistry
from ml_training_template.application import TrainApplication

# Models
@ModelRegistry.register()
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

# Model Container
@ModelContainerRegistry.register()
class MNISTModelContainer(BaseModelContainer):
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

if __name__ == "__main__":
    train_app = TrainApplication()
    train_app.run()
>>>
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs

  | Name  | Type       | Params
-------------------------------------
0 | model | MnistModel | 21.8 K
-------------------------------------
21.8 K    Trainable params
0         Non-trainable params
21.8 K    Total params
0.087     Total estimated model params size (MB)
Epoch 0:  12%|██████                                             | 33/275 [00:01<00:10, 22.75it/s, loss=2.31, v_num=9]
...
```

## Expected difficulties

1. Generalized structure that satisfies various architectures such as Hugging Face, Transformer, Encoder-Decoder, etc.
2. Generalized structure that can perform various tasks such as vision, nlp, and voice processing

## Installation

### For Developments

```bash
poetry install
```

### Build Packages

```bash
poetry build
python3 -m pip install ./dist/ml_training_template-0.0.1.dev0-py3-none-any.whl
```

## Runnable Examples

```bash
export PYATHONPATH="[PROJECT_DIR]/example/mnist"
python3 train.py
```

## Architecture

대부분의 인터페이스는 PyTorch Lightning의 인터페이스를 따른다.
PyTorch Lightning을 한단계 더 추상화하여
