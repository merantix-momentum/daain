"""
These classes in combination (check the data.py file on how it is being used) allow training on multiple differently
sized training sets.

Everything should be somewhat clear as long as you remember that each items in a batch have to have the same dimensions.
Thus we need to select the items accordingly. On the other hand we need to make sure that all items are selected.
"""
from typing import Optional, Sized

import torch
from torch.utils.data import RandomSampler, SequentialSampler


class ConcatTupleDataset(torch.utils.data.Dataset):
    """Concatenates multiple datasets. Each iteration will yield a tuple of items, one item in the tuple for each
    dataset to be concatenated."""

    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        # in theory one could wrap them differently (and consider each size individually) but this just makes
        # everything much, much easier
        return min(len(d) for d in self.datasets)


class MultipleSequentialSampler(SequentialSampler):
    """Yields the same items multiple times."""

    def __init__(self, data_source: Optional[Sized], num_times) -> None:
        super().__init__(data_source)
        self.num_times = num_times
        self.data_source = data_source

    def __iter__(self):
        for _ in range(self.num_times):
            yield from super().__iter__()

    def __len__(self):
        return len(self.data_source) * self.num_times


class MultipleRandomSampler(RandomSampler):
    """Samples with a fixed number of replacements"""

    def __init__(self, data_source: Optional[Sized], num_times) -> None:
        super().__init__(data_source)
        self.num_times = num_times

    def __iter__(self):
        for _ in range(self.num_times):
            yield from super().__iter__()

    def __len__(self):
        return len(self.data_source) * self.num_times


class DatasetCollate:
    """Closure to call all datasets equally. a functional closure doesn't work because of pickling issues"""

    def __init__(self):
        self.current_dataset_idx = 0

    def __call__(self, to_batch):
        r = torch.utils.data._utils.collate.default_collate([el[self.current_dataset_idx] for el in to_batch])
        self.current_dataset_idx = (self.current_dataset_idx + 1) % len(to_batch[0])
        return r
