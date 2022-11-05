from dependency_injector import containers, providers
from pytorch_lightning.callbacks import ModelCheckpoint

from ...core.patterns.registry import ModelCheckpointRegistry


def register_default_modelcheckpoint():
    ModelCheckpointRegistry.register(ModelCheckpoint)


class ModelCheckpoint(containers.DeclarativeContainer):
    register_default_modelcheckpoint()

    config = providers.Configuration()
    instance = ModelCheckpointRegistry.get(config.name)
    checkpoint = providers.Singleton(
        instance,
        config.params,
    )
