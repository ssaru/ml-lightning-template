from dependency_injector import containers, providers

from app.core.patterns.registry import DataLoaderRegistry, DatasetRegistry


class DataLoader(containers.DeclarativeContainer):
    config = providers.Configuration()

    dataset_instance = DatasetRegistry.get(config.dataset.name)
    dataloder_instance = DataLoaderRegistry.get(config.dataloder.name)

    dataset = providers.Singleton(
        dataset_instance,
        config.dataset.params
    )

    config.dataloader.params.update({"dataset": dataset})
    dataloader = providers.Singleton(
        dataloder_instance,
        config.dataloader.params,
    )
