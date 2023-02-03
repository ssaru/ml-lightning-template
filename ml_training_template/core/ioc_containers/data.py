from typing import Dict

from ml_training_template.core.patterns.registry import (
    DataLoaderRegistry,
    DatasetRegistry,
    TransformRegistry,
)

from .base import BaseIoCContainer


class DataIoCContainer(BaseIoCContainer):
    def __init__(self,
                 dataset_name: str,
                 dataset_params: Dict,
                 dataloader_name: str,
                 dataloader_params: Dict):
        dataset = self.get_dataset(name=dataset_name, params=dataset_params)
        self.dataloader = self.get_dataloader(name=dataloader_name,
                                              params=dataloader_params,
                                              dataset=dataset)

    def get(self):
        return self.dataloader

    def get_dataset(self, name, params):
        transform = self._instanciate_transform(
            transform_params=params.pop("transform"))
        params.update({"transform": transform})

        dataset_cls = DatasetRegistry.get(name)
        dataset = dataset_cls(**params)
        return dataset

    def _instanciate_transform(self, transform_params: Dict):
        transform_name = transform_params.get("name")
        transform_params = transform_params.get("params", {})
        transform_cls = TransformRegistry.get(transform_name)
        transform = transform_cls(**transform_params)
        return transform

    def get_dataloader(self, name, params, dataset):
        dataloader_cls = DataLoaderRegistry.get(name)
        params.update({"dataset": dataset})
        dataloader = dataloader_cls(**params)
        return dataloader
