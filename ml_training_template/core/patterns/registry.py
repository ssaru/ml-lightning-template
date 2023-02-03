from typing import Any, Dict, Iterable, Iterator, Tuple

from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets as vision_datasets
from torchvision import transforms

from ml_training_template.core.interfaces import (
    BaseDataLoader,
    BaseModelContainer,
    BaseTrainer,
)


class Registry(Iterable[Tuple[str, Any]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj: Any = None) -> Any:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            ImportWarning(
                f"No object named '{name}' found in '{self._name}' registry!")
            return None
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        items = "\n".join([f'\t{k}: {v}' for k, v in self._obj_map.items()])
        return f"Registry of {self._name}:\n{items}"

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__


TransformRegistry = Registry("TRANSFORM")

DatasetRegistry = Registry("DATASET")
DataLoaderRegistry = Registry("DATALOADER")

ModelRegistry = Registry("MODEL")
ModelContainerRegistry = Registry("MODEL_CONTAINER")
ModelCheckpointRegistry = Registry("MODEL_CHECKPOINT")

OptimizerRegistry = Registry("OPTIMIZER")
SchedulerRegistry = Registry("SCHEDULER")

TrainerRegistry = Registry("TRAINER")


def register_default_optimizer():
    for k, v in optim.__dict__.items():
        if not (k.startswith("__") or
                k.startswith("_") or
                k.islower() or
                k in ("Optimizer")):
            OptimizerRegistry.register(v)


def register_default_scheduler():
    for k, v in lr_scheduler.__dict__.items():
        if not (k.startswith("__") or
                k.startswith("_") or
                k.islower() or
                k.isupper() or
                k in ("Optimizer", "Counter", "EPOCH_DEPRECATION_WARNING")):
            SchedulerRegistry.register(v)


def register_default_datasets():
    for k, v in vision_datasets.__dict__.items():
        if not (k.startswith("__") or
                k.startswith("_") or
                k.islower() or
                k in ("ImageFolder", "DatasetFolder")):
            DatasetRegistry.register(v)


def register_default_modelcheckpoint():
    ModelCheckpointRegistry.register(ModelCheckpoint)


def register_default_trainer():
    TrainerRegistry.register(BaseTrainer)


def register_default_dataloader():
    DataLoaderRegistry.register(BaseDataLoader)


def register_default_model_container():
    ModelContainerRegistry.register(BaseModelContainer)


def register_default_transform():
    for k, v in transforms.__dict__.items():
        if not(
            k.startswith("__") or k.startswith("_") or k.islower()
            or
            k
            in (
                "Compose", "autoaugment", "functional", "functional_pil",
                "functional_tensor", "transforms")):
            TransformRegistry.register(v)


register_default_optimizer()
register_default_scheduler()
register_default_datasets()
register_default_modelcheckpoint()
register_default_dataloader()
register_default_trainer()
register_default_transform()


__all__ = ["DatasetRegistry", "DataLoaderRegistry", "ModelRegistry",
           "ModelContainerRegistry", "ModelCheckpointRegistry",
           "OptimizerRegistry", "SchedulerRegistry", "TrainerRegistry"]
