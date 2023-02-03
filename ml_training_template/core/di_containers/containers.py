from dependency_injector import containers, providers

from ml_training_template.config import AppConfig
from ml_training_template.core.ioc_containers import (
    DataIoCContainer,
    ModelCheckPointIoCContainer,
    ModelIoCContainer,
    TrainerIoCContainer,
)


class Model(containers.DeclarativeContainer):
    config = providers.Configuration()

    container = providers.Singleton(
        ModelIoCContainer,
        model_name=config.model.name,
        model_params=config.model.params,
        optimizer_name=config.optimizer.name,
        optimizer_params=config.optimizer.params,
        scheduler_name=config.scheduler.name,
        scheduler_params=config.scheduler.params,
        container_name=config.name,
        container_params=config.params
    )


class Trainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    executor = providers.Singleton(
        TrainerIoCContainer,
        trainer_name=config.name,
        trainer_params=config.params
    )


class Data(containers.DeclarativeContainer):
    config = providers.Configuration()

    loader = providers.Singleton(
        DataIoCContainer,
        dataset_name=config.dataset.name,
        dataset_params=config.dataset.params,
        dataloader_name=config.dataloader.name,
        dataloader_params=config.dataloader.params
    )


class ModelCheckpoint(containers.DeclarativeContainer):
    config = providers.Configuration()

    checkpoint = providers.Singleton(
        ModelCheckPointIoCContainer,
        model_checkpoint_name=config.name,
        model_checkpoint_params=config.params
    )


class Training(containers.DeclarativeContainer):
    config = providers.Configuration()
    config.from_pydantic(AppConfig.MODEL)

    train_data = providers.Container(
        Data,
        config=config.data.train
    )

    valid_data = providers.Container(
        Data,
        config=config.data.valid
    )

    test_data = providers.Container(
        Data,
        config=config.data.test
    )

    model = providers.Container(
        Model,
        config=config.container
    )

    train = providers.Container(
        Trainer,
        config=config.trainer
    )
