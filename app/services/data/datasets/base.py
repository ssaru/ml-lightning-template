
from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()
