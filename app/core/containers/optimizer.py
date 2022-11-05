from dependency_injector import containers, providers

from ...core.patterns.registry import OptimizerRegistry


class Optimizer(containers.DeclarativeContainer):
    config = providers.Configuration()
    instance = OptimizerRegistry.get(config.name)
    optimizer = providers.Singleton(
        instance,
        config.params,
    )
