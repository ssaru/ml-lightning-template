from abc import ABC
from typing import Dict

from ml_training_template.core.patterns.registry import (
    DataLoaderRegistry,
    DatasetRegistry,
)


class DataIoCContainer(ABC):
    def __init__(self,
                 dataset_name: str,
                 dataset_params: Dict,
                 dataloader_name: str,
                 dataloader_params: Dict):
        dataset = self.get_dataset(name=dataset_name, params=dataset_params)
        self.dataloader = self.get_dataloader(name=dataloader_name,
                                              params=dataloader_params,
                                              dataset=dataset)

    def get_dataset(self, name, params):
        dataset_cls = DatasetRegistry.get(name)
        dataset = dataset_cls(**params)
        return dataset

    def get_dataloader(self, name, params, dataset):
        dataloader_cls = DataLoaderRegistry.get(name)
        params.update({"dataset": dataset})
        dataloader = dataloader_cls(**params)
        return dataloader
