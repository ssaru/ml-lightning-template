from typing import Type

from dependency_injector import containers, providers

from app.core.interfaces.models import BaseModel
from app.core.interfaces.optimizer import BaseOptimizer
from app.core.interfaces.scheduler import BaseScheduler
from app.core.patterns.registry import (
    ModelContainerRegistry,
    ModelRegistry,
    OptimizerRegistry,
    SchedulerRegistry,
)


class ModelContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    model_instance = ModelRegistry.get(config.model.name)

    optimizer_instance = OptimizerRegistry.get(config.train.optimizer.name)
    scheduler_instance = SchedulerRegistry.get(
        config.train.optimizer.scheduler.name)

    container_instance = ModelContainerRegistry.get(
        config.model_container.name)

    model: Type["BaseModel"] = providers.Singleton(
        model_instance,
        config.model.params,
    )

    config.train.optimizer.params.update({"params": model.parameters()})
    optimizer: Type["BaseOptimizer"] = providers.Singleton(
        optimizer_instance,
        config.train.optimizer.params,
    )
    config.model_container.params.update({"optimizer": optimizer})

    if scheduler_instance:
        config.train.optimizer.scheduler.params.update(
            {"optimizer": optimizer})
        scheduler: Type["BaseScheduler"] = providers.Singleton(
            scheduler_instance,
            config.train.optimizer.scheduler.params,
        )
        config.model_container.params.update({"scheduler": scheduler})

    config.model_container.params.update({"model": model})
    container = providers.Singleton(
        container_instance,
        config.model_container.params,
    )
