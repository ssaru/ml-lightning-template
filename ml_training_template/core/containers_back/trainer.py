from dependency_injector import containers, providers

from ml_training_template.core.patterns.registry import (
    ModelCheckpointRegistry,
    TrainerRegistry,
)


class Trainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    model_checkpoint_instance = ModelCheckpointRegistry.get(
        config.model_checkpoint.name)
    trainer_instance = TrainerRegistry.get(config.name)

    model_checkpoint = providers.Singleton(
        model_checkpoint_instance,
        config.model_checkpoint.params,
    )

    config.params.update({"checkpoint_callback": model_checkpoint})
    trainer = providers.Container(
        trainer_instance,
        config=config.params
    )
