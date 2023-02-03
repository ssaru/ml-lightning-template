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

        print("TRAIN")
        print(f"config: {app.config()}")
        app.core.init_resources()
        app.wire(modules=[__name__])

    @inject
    def run(self,
            train_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.train_dataloader],
            valid_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.valid_dataloader],
            test_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.test_dataloader],
            model_container: Type["BaseModelContainer"] = Provide[TrainExecutor.model_container],
            trainer: Type["BaseTrainer"] = Provide[TrainExecutor.trainer]
            ) -> None:
        print(f"dataloader: {train_dataloader}")
        print(f"{dir(train_dataloader)}")
        print("====================================\n\n")
        print(f"valid_dataloader: {valid_dataloader}")
        print(f"{dir(valid_dataloader)}")
        print("====================================\n\n")
        print(f"test_dataloader: {test_dataloader}")
        print(f"{dir(test_dataloader)}")
        print("====================================\n\n")
        print(f"model_container: {model_container}")
        print(f"{dir(model_container)}")
        print("====================================\n\n")
        print(f"trainer: {trainer}")
        print(f"{dir(trainer)}")
        print("====================================\n\n")

        trainer.fit(model=model_container,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=valid_dataloader)
        trainer.test(model=model_container,
                     dataloaders=test_dataloader)


@inject
def main(train_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.train_dataloader],
         valid_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.valid_dataloader],
         test_dataloader: Type["BaseDataLoader"] = Provide[TrainExecutor.test_dataloader],
         model_container: Type["BaseModelContainer"] = Provide[TrainExecutor.model_container],
         trainer: Type["BaseTrainer"] = Provide[TrainExecutor.trainer]
         ) -> None:
    print(f"dataloader: {train_dataloader}")
    print(f"{dir(train_dataloader)}")
    print("====================================\n\n")
    print(f"valid_dataloader: {valid_dataloader}")
    print(f"{dir(valid_dataloader)}")
    print("====================================\n\n")
    print(f"test_dataloader: {test_dataloader}")
    print(f"{dir(test_dataloader)}")
    print("====================================\n\n")
    print(f"model_container: {model_container}")
    print(f"{dir(model_container)}")
    print("====================================\n\n")
    print(f"trainer: {trainer}")
    print(f"{dir(trainer)}")
    print("====================================\n\n")
    trainer.fit(model=model_container,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader)
    trainer.test(model=model_container,
                 dataloaders=test_dataloader)


if __name__ == "__main__":
    container = TrainExecutor()
    container.core.init_resources()
    container.wire(modules=[__name__])
    main()
