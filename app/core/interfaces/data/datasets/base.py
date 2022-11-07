
from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
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
