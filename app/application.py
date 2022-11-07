from typing import Type

from dependency_injector.wiring import Provide, inject

from app.core.containers import TrainExecutor
from app.core.interfaces.data.dataloaders import BaseDataLoader
from app.core.interfaces.models.containers import BaseModelContainer
from app.core.interfaces.trainer import BaseTrainer


class Application:

    def __init__(self):
        pass

    def train(self):
        train_application = TrainExecutor()
        train_application.core.init_resources()
        train_application.wire(modules=[__name__])
        self._train()

    @inject
    def _train(self,
               train_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.train_dataloader],
               valid_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.valid_dataloader],
               test_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.test_dataloader],
               model_container: Type["BaseModelContainer"] = Provide[TrainExecutor.model_container],
               trainer: Type["BaseTrainer"] = Provide[TrainExecutor.trainer]
               ) -> None:
        trainer.fit(model=model_container,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=valid_dataloader,
                    test_dataloader=test_dataloader)


if __name__ == "__main__":
    app = Application()
    app.train()
