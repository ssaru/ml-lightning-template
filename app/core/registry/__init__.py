from .registry import Registry

DatasetRegistry = Registry("DATASET")
DataLoaderRegistry = Registry("DATALOADER")
BackBoneRegistry = Registry("BACKBONE")
OptimizerRegistry = Registry("OPTIMIZER")
SchedulerRegistry = Registry("SCHEDULER")
ModelCheckpointRegistry = Registry("MODEL_CHECKPOINT")
