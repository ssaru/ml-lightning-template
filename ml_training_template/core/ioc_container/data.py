from abc import ABC
from typing import Dict

from ml_training_template.core.patterns.registry import (
    DataLoaderRegistry,
    DatasetRegistry,
    TransformRegistry,
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
        transform_dict = params.pop("transform")
        transform_name = transform_dict.get("name")
        transform_params = transform_dict.get("params", {})
        print(f"transforms_name: {transform_name}")
        print(f"transforms_params: {transform_params}")
        transform_cls = TransformRegistry.get(transform_name)
        transforms = transform_cls(**transform_params)
        params.update({"transform": transforms})

        dataset_cls = DatasetRegistry.get(name)
        dataset = dataset_cls(**params)
        return dataset

    def get_dataloader(self, name, params, dataset):
        dataloader_cls = DataLoaderRegistry.get(name)
        params.update({"dataset": dataset})
        dataloader = dataloader_cls(**params)
        return dataloader
