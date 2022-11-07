from abc import ABC, abstractmethod

from ..models.containers.base import BaseModelContainer


class Trainer(ABC):

    @abstractmethod
    def fit(model: BaseModelContainer):
        pass
