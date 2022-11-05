from .registry import Registry

DatasetRegistry = Registry("DATASET")
DataLoaderRegistry = Registry("DATALOADER")

ModelRegistry = Registry("MODEL")
ModelContainerRegistry = Registry("MODEL_CONTAINER")
ModelCheckpointRegistry = Registry("MODEL_CHECKPOINT")

OptimizerRegistry = Registry("OPTIMIZER")
SchedulerRegistry = Registry("SCHEDULER")
