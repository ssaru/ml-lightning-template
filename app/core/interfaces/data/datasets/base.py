
from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from ....patterns.registry import DatasetRegistry

# NOTE. BaseDataset class는 추상 클래스라 Registry에 등록하지 않음


class BaseDataset(Dataset, ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def download(self):
        raise NotImplementedError()

    @abstractmethod
    def optimize_dtypes(self):
        raise NotImplementedError()

    @abstractmethod
    def validate(self):
        raise NotImplementedError()
