from typing import Type

from dependency_injector.wiring import Provide, inject

from ml_training_template.core.di_containers import Training
from ml_training_template.core.ioc_containers import (
    DataIoCContainer,
    ModelIoCContainer,
    TrainerIoCContainer,
)


class TrainApplication:
    def __init__(self):
        app = Training()
        app.wire(packages=[__name__])

    @inject
    def run(self,
            train_data_container: Type["DataIoCContainer"] = Provide[Training.train_data.loader],
            valid_data_container: Type["DataIoCContainer"] = Provide[Training.valid_data.loader],
            test_data_container: Type["DataIoCContainer"] = Provide[Training.test_data.loader],
            model_container: Type["ModelIoCContainer"] = Provide[Training.model.container.container],
            trainer_container: Type["TrainerIoCContainer"] = Provide[Training.train.executor]
            ) -> None:
        model = model_container.get()
        trainer = trainer_container.get()
        train_dataloaders = train_data_container.get()
        valid_dataloaders = valid_data_container.get()
        test_dataloaders = test_data_container.get()

        trainer.fit(model=model,
                    train_dataloaders=train_dataloaders,
                    val_dataloaders=valid_dataloaders)
        trainer.test(model=model,
                     dataloaders=test_dataloaders)
