import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from omegaconf import OmegaConf
from pydantic import BaseSettings

from ml_training_template.config.utils import load_config
from ml_training_template.core.patterns import Singleton


class BaseConfig(BaseSettings):
    @classmethod
    def of(cls, params: Dict):
        return cls.parse_obj(OmegaConf.to_container(OmegaConf.create(params)))

# Common Configurations


class Logger(BaseConfig):
    level: str = "INFO"
    max_len: int = 500
    handler: List[str]


class Dataset(BaseConfig):
    name: str
    # NOTE. Dataset Class의 params는 일반화하기 어려워 Dict로 정의
    params: Dict


# Modeling Configurations
class DataLoader(BaseConfig):
    name: str = "BaseDataLoader"
    # NOTE. DataLoader의 params는 일반화하기 어려워 Dict로 정의
    # PyTorch에서 정의하는 DataLoader params의 기본 구성은 아래와 같음
    # `dataset`은 application 로드 시, 동적으로 삽입
    # {
    #   "dataset": None,
    #   "batch_size": 1,
    #   "shuffle": None,
    #   "num_workers": 0,
    #   "drop_last": False
    # }
    params: Dict = {}


class DataModule(BaseConfig):
    dataset: Dataset
    dataloader: DataLoader


class Data(BaseConfig):
    train: DataModule
    valid: Optional[DataModule]
    test: Optional[DataModule]


class Model(BaseConfig):
    name: str
    params: Optional[Dict] = {}


class Scheduler(BaseConfig):
    name: Union[str, None] = None
    params: Optional[Dict] = {}


class Optimizer(BaseConfig):
    name: str = "SGD"
    params: Optional[Dict] = {"lr": 1e-2, "momentum": 0.9}


class ModelContainer(BaseConfig):
    name: str
    params: Optional[Dict] = {}
    model: Model
    optimizer: Optimizer = Optimizer()
    scheduler: Optional[Scheduler] = Scheduler()


class ModelCheckpoint(BaseConfig):
    name: str = "ModelCheckpoint"
    params: Optional[Dict] = {"dirpath": "./outputs", "filename": "model"}


class Trainer(BaseConfig):
    name: str = "BaseTrainer"
    params: Dict = {}


class Model(BaseConfig):
    data: Data
    container: ModelContainer
    trainer: Trainer
    model_checkpoint: Optional[ModelCheckpoint] = ModelCheckpoint()


# Application Configurations
class AppConfig(Singleton):
    PROJECT_NAME: str
    LOGGER: Logger

    DEBUG: bool

    ROOT_PATH: Path
    RESOURCE_PATH: Path

    MODEL: Model

    def __init__(self):
        self.ROOT_PATH = Path(os.getcwd())
        self.RESOURCE_PATH = self.ROOT_PATH / "resources"

        config = load_config(path=self.RESOURCE_PATH)
        from pprint import pprint
        pprint(config)

        self.PROJECT_NAME = config["app"]["project_name"]
        self.MAIN_LOG = Logger.of(config["log"]["main_log"])
        self.MODEL = Model.of(config["modeling"])
