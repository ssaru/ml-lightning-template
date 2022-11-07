from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets as vision_datasets

from app.core.interfaces.trainer import LightningTrainer
from app.core.patterns.registry import Registry

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
    TrainerRegistry.register(LightningTrainer)


register_default_optimizer()
register_default_scheduler()
register_default_datasets()
register_default_modelcheckpoint()
