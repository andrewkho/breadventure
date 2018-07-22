import logging
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, Sampler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RecipesDataset(Dataset):
    """
    Holds a 1D LongTensor, each element indexes into a word in vocab

    """
    def __init__(self, data: torch.LongTensor):
        """
        Args:
            root_dir (string): Directory with recipes {train,valid,test}.txt
        """
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class SkipSampler(Sampler):
    def __init__(self, data_source, batch_size: int, bptt: int, rand: bool):
        super().__init__(data_source)
        self.data_source = data_source
        self._batch_size = batch_size
        self._skip = len(self.data_source) // self._batch_size
        self._trunc_sz = self._skip * batch_size
        self._rand = rand
        self._bptt = bptt

        self._shrink = 0
        if self._rand:
            self._shrink = bptt*batch_size

    def __iter__(self):
        start = 0
        if self._rand:
            start = random.choice(range(self._bptt))
        for i in range(start, self._skip):
            for j in range(self._batch_size):
                yield j * self._skip + i

    def __len__(self):
        return self._trunc_sz - self._shrink


def collate_batch(batch_sz):
    def _collate_batch(batch):
        data = torch.stack(batch).view(-1, batch_sz)
        return data[:-1, :], data[1:, :].view(-1)

    return _collate_batch


def create_dataloader(dataset: Dataset,
                      batch_size: int,
                      bptt: int,
                      rand: bool) -> DataLoader:
    return DataLoader(dataset,
                      batch_sampler=BatchSampler(
                          SkipSampler(dataset,
                                      batch_size=batch_size,
                                      bptt=bptt,
                                      rand=rand),
                          batch_size=batch_size * (bptt + 1),
                          drop_last=True),
                      collate_fn=collate_batch(batch_size))
