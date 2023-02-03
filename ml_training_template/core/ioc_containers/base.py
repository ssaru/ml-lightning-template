from abc import ABC, abstractmethod


class BaseIoCContainer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self):
        raise NotImplementedError()
