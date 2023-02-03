from abc import ABC
from typing import Dict, Optional, Type

from ml_training_template.core.interfaces.models import BaseModel
from ml_training_template.core.interfaces.models.containers import (
    BaseModelContainer,
)
from ml_training_template.core.interfaces.optimizer import BaseOptimizer
from ml_training_template.core.interfaces.scheduler import BaseScheduler
from ml_training_template.core.patterns.registry import (
    ModelContainerRegistry,
    ModelRegistry,
    OptimizerRegistry,
    SchedulerRegistry,
)


class ModelIoCContainer(ABC):
    def __init__(self,
                 model_name: str,
                 model_params: Dict,
                 optimizer_name: str,
                 optimizer_params: Dict,
                 scheduler_name: str,
                 scheduler_params: Dict,
                 container_name: str,
                 container_params: Dict):
        model: Type["BaseModel"] = self.get_model(
            name=model_name, params=model_params)
        optimizer: Type["BaseOptimizer"] = self.get_optimizer(
            model=model, name=optimizer_name, params=optimizer_params)
        scheduler: Optional[Type["BaseScheduler"]] = self.get_scheduler(
            name=scheduler_name, params=scheduler_params, optimizer=optimizer)
        self.container: Type["BaseModelContainer"] = self.get_model_container(
            name=container_name, params=container_params, model=model,
            optimizer=optimizer, scheduler=scheduler)

    def get_model(self, name: str, params: Dict):
        model_cls = ModelRegistry.get(name)
        model = model_cls(**params)
        return model

    def get_optimizer(self, model: Type[BaseModel], name: str, params: Dict):
        optimizer_cls = OptimizerRegistry.get(name)
        params.update({"params": model.parameters()})
        optimizer = optimizer_cls(**params)
        return optimizer

    def get_scheduler(self, name: str, params: Dict,
                      optimizer: Type["BaseOptimizer"]):
        if name is None:
            return None
        scheduler_cls = SchedulerRegistry.get(name)
        scheduler = scheduler_cls(optimizer=optimizer, **params)
        return scheduler

    def get_model_container(self,
                            name: str,
                            model: Type["BaseModel"],
                            optimizer: Type["BaseOptimizer"],
                            scheduler: Type["BaseScheduler"],
                            params: Dict,):
        model_container_cls = ModelContainerRegistry.get(name)
        model_container = model_container_cls(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            **params
        )
        return model_container
