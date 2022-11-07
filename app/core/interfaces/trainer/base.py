from abc import ABC
from typing import Any

from pytorch_lightning import Trainer

from app.core.interfaces.models.containers.base import BaseModelContainer


class BaseTrainer(ABC, Trainer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    trainer = BaseTrainer()
    print(dir(trainer))
