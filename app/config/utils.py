import logging
from typing import Dict

import yaml

logging.getLogger(__name__).addHandler(logging.NullHandler())


def merge(active_config: Dict, default_config: Dict):
    """active_config을 default_config에 오버라이딩하는 방향으로 두 config을 병합한다"""
    if isinstance(active_config, dict) and isinstance(default_config, dict):
        for k, v in default_config.items():
            if k not in active_config:
                active_config[k] = v
            else:
                active_config[k] = merge(active_config[k], v)
    elif isinstance(active_config, dict) and default_config is None:
        return active_config
    elif isinstance(default_config, dict) and active_config is None:
        return default_config
    return active_config


def load_config_from(filepath: str):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
        if not config:
            return {}
        return config


def load_config(profile: str, path: str):
    default_conf = load_config_from(filepath=f"{path}/application.yaml")
    active_conf = load_config_from(
        filepath=f"{path}/application-{profile}.yaml")

    return merge(active=active_conf, default=default_conf)
