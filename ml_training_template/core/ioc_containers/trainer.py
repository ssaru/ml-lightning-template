from ml_training_template.core.patterns.registry import TrainerRegistry

from .base import BaseIoCContainer


class TrainerIoCContainer(BaseIoCContainer):
    def __init__(self,
                 trainer_name,
                 trainer_params):
        self.trainer = self.get_trainer(name=trainer_name,
                                        params=trainer_params)

    def get(self):
        return self.trainer

    def get_trainer(self, name, params):
        trainer_cls = TrainerRegistry.get(name)
        trainer = trainer_cls(**params)
        return trainer
