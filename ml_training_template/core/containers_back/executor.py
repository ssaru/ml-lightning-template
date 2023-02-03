from dependency_injector import containers, providers

from ml_training_template.config import AppConfig


class TrainExecutor(containers.DeclarativeContainer):
    from ml_training_template.core.containers.dataloader import DataLoader
    from ml_training_template.core.containers.model_container import (
        ModelContainer,
    )
    from ml_training_template.core.containers.trainer import Trainer

    config = providers.Configuration()
    config.from_pydantic(AppConfig.MODEL)
    print(config)

    train_dataloader = providers.Container(
        DataLoader,
        config=config.data.train
    )

    valid_dataloader = providers.Container(
        DataLoader,
        config=config.data.valid
    )

    test_dataloader = providers.Container(
        DataLoader,
        config=config.data.test
    )

    model_container = providers.Container(
        ModelContainer,
        config=config
    )

    trainer = providers.Container(
        Trainer,
        config=config.trainer
    )
