from abc import ABC
from typing import Dict

from ml_training_template.core.patterns.registry import ModelCheckpointRegistry


class ModelCheckPointIoCContainer(ABC):
    def __init__(self,
                 model_checkpoint_name: str,
                 model_checkpoint_params: Dict):
        self.model_checkpoint = self.get_model_checkpoint(
            name=model_checkpoint_name, params=model_checkpoint_params)

    def get_model_checkpoint(self, name, params):
        model_checkpoint_cls = ModelCheckpointRegistry.get(name)
        model_checkpoint = model_checkpoint_cls(**params)
        return model_checkpoint
