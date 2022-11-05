from dependency_injector import containers, providers

from ...core.patterns.registry import ModelContainerRegistry


class ModelContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    instance = ModelContainerRegistry.get(config.name)
    container = providers.Singleton(
        instance,
        config.params,
    )
