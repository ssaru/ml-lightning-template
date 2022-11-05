from typing import Optional, TypeVar

from torch.utils.data import DataLoader, Dataset

from ....patterns.registry import DataLoaderRegistry

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


# NOTE. BaseDataLoader는 추상클래스이기도 하면서, 그대로 사용할 수 있기에 Registry에 등록
@DataLoaderRegistry.register
class BaseDataLoader(DataLoader):
    """It follows PyTorch's DataLoader Interface.

    NOTE. This is subject to change in the future.
    """

    def __init__(self,
                 dataset: Dataset[T_co],
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None,
                 num_workers: int = 0,
                 drop_last: bool = False):
        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         drop_last=drop_last)
