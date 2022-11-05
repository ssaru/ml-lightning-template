from dependency_injector import containers, providers
from torchvision import datasets as vision_datasets

from ...core.patterns.registry import DatasetRegistry


def register_default_optimizer():
    for k, v in vision_datasets.__dict__.items():
        if not (k.startswith("__") or
                k.startswith("_") or
                k.islower() or
                k in ("ImageFolder", "DatasetFolder")):
            DatasetRegistry.register(v)


class Dataset(containers.DeclarativeContainer):
    config = providers.Configuration()
    instance = DatasetRegistry.get(config.name)
    dataset = providers.Singleton(
        instance,
        config.params,
    )
