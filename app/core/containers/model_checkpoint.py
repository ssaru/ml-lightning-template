from dependency_injector import containers, providers

from ...core.patterns.registry import ModelCheckpointRegistry


class ModelCheckpoint(containers.DeclarativeContainer):
    config = providers.Configuration()
    instance = ModelCheckpointRegistry.get(config.name)
    checkpoint = providers.Singleton(
        instance,
        config.params,
    )
