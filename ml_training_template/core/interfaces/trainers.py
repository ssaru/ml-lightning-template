from abc import ABC
from typing import Any

from pytorch_lightning import Trainer


class BaseTrainer(ABC, Trainer):
    def __init__(self,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
