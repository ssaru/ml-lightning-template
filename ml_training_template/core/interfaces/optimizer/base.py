from abc import ABC
from typing import Any

from torch.optim import Optimizer


class BaseOptimizer(ABC, Optimizer):
    """
    Optimizer에 대해 뚜렷한 인터페이스가 없음. 필요 시 추가
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
