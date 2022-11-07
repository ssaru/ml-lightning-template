from dependency_injector import containers, providers

from app.config import AppConfig
from app.core.containers.dataloader import DataLoader
from app.core.containers.model_container import ModelContainer
from app.core.containers.trainer import Trainer


class TrainExecutor(containers.DeclarativeContainer):
    config = providers.Configuration()
    config.from_pydantic(AppConfig)

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
