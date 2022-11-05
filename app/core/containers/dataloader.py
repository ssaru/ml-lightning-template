from dependency_injector import containers, providers

from ...core.interfaces.data.datasets import BaseDataset
from ...core.patterns.registry import DataLoaderRegistry


class DataLoader(containers.DeclarativeContainer):
    config = providers.Configuration()
    instance = DataLoaderRegistry.get(config.name)
    dataloader = providers.Singleton(
        instance,
        config.params,
    )
