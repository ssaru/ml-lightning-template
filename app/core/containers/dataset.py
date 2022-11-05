from dependency_injector import containers, providers

from ...core.patterns.registry import DatasetRegistry


class Dataset(containers.DeclarativeContainer):
    config = providers.Configuration()
    instance = DatasetRegistry.get(config.name)
    dataset = providers.Singleton(
        instance,
        config.params,
    )
