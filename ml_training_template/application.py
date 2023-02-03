from typing import Type

from dependency_injector.wiring import Provide, inject

from ml_training_template.core.containers import TrainExecutor
from ml_training_template.core.interfaces.data.dataloaders import (
    BaseDataLoader,
)
from ml_training_template.core.interfaces.models.containers import (
    BaseModelContainer,
)
from ml_training_template.core.interfaces.trainer import BaseTrainer


class TrainApplication:
    def __init__(self):
        app = TrainExecutor()
        app.wire(modules=["ml_training_template"])

    @inject
    def run(self,
            train_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.train_dataloader],
            valid_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.valid_dataloader],
            test_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.test_dataloader],
            model_container: Type["BaseModelContainer"] = Provide[TrainExecutor.model_container],
            trainer: Type["BaseTrainer"] = Provide[TrainExecutor.trainer]
            ) -> None:

        trainer.fit(model=model_container,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=valid_dataloader)
        trainer.test(model=model_container,
                     dataloaders=test_dataloader)


@inject
def main(train_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.train_dataloader.dataloader],
         valid_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.valid_dataloader.dataloader],
         test_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.test_dataloader.dataloader],
         model_container: Type["BaseModelContainer"] = Provide[TrainExecutor.model_container.model_container],
         trainer: Type["BaseTrainer"] = Provide[TrainExecutor.trainer.trainer]
         ) -> None:
    trainer.trainer.fit(model=model_container.container,
                        train_dataloaders=train_dataloader.dataloader,
                        val_dataloaders=valid_dataloader.dataloader)
    trainer.trainer.test(model=model_container.container,
                         dataloaders=test_dataloader.dataloader)


if __name__ == "__main__":
    container = TrainExecutor()
    container.core.init_resources()
    container.wire(modules=[__name__])
    main()
