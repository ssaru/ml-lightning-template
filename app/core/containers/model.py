from dependency_injector import containers, providers

from ...core.patterns.registry import ModelRegistry


class Model(containers.DeclarativeContainer):
    config = providers.Configuration()
    instance = ModelRegistry.get(config.name)
    model = providers.Singleton(
        instance,
        config.params,
    )
