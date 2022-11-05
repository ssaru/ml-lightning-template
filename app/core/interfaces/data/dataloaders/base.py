from typing import Optional, TypeVar

from torch.utils.data import DataLoader, Dataset

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


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
