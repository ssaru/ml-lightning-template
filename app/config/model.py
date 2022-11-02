import os
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf
from pydantic import BaseModel

from ..core.patterns import Singleton
from .utils import load_config


class BaseConfig(BaseModel):
    @classmethod
    def of(cls, params: Dict):
        return cls.parse_obj(OmegaConf.to_container(OmegaConf.create(params)))


class Logger(BaseConfig):
    level: str = "INFO"
    max_len: int = 500
    handler: List[str]


class AppConfig(Singleton):
    PROJECT_NAME: str
    LOGGER: Logger

    PROFILE: str
    DEBUG: bool

    LIVE: bool
    DEV: bool
    LOCAL: bool

    ROOT_PATH: Path
    RESOURCE_PATH: Path

    def __init__(self):
        self.PROFILE = os.getenv("PROFILE", "local")
        self.ROOT_PATH = Path(__file__).parent.parent.parent.absolute()
        self.RESOURCE_PATH = self.ROOT_PATH / "resources"

        config = load_config(
            profile=self.PROFILE, path=os.path.join(
                self.RESOURCE_PATH, "config"))

        self.PROJECT_NAME = config["app"]["project_name"]
        self.MAIN_LOG = Logger.of(config["log"]["main_log"])

        self.DEBUG = (self.PROFILE != "live")
        self.LIVE = (self.PROFILE == "live")
        self.LOCAL = (self.PROFILE == "local")
        self.DEV = (self.PROFILE == "dev")
