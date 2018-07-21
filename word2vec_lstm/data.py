import os
import logging

import torch
from torch.utils.data import Dataset

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


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path: str, device: str):
        logger.info('reading train/valid/test')
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt')).to(device)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt')).to(device)
        self.test = self.tokenize(os.path.join(path, 'test.txt')).to(device)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

