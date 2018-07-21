import os
import logging
from typing import List

import torch
from gensim.models.keyedvectors import KeyedVectors


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    def __init__(self, path: str):
        logger.info('reading train/valid/test')
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

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


class Corpus2(object):
    def __init__(self,
                 path: str,
                 w2v_path: str= 'data/GoogleNews-vectors-negative300.bin'):
        logger.info('reading word2vec trained model')
        self.w2c_model = KeyedVectors.load_word2vec_format(w2v_path,
                                                           binary=True)

        self.dictionary = Dictionary()
        logger.info('reading train/valid/test')

        self.train_tokens = self.tokenize(os.path.join(path, 'train.txt'))
        logger.info(f'read {len(self.train_tokens)} tokens')
        self.valid_tokens = self.tokenize(os.path.join(path, 'valid.txt'))
        logger.info(f'read {len(self.valid_tokens)} tokens')
        self.test_tokens = self.tokenize(os.path.join(path, 'test.txt'))
        logger.info(f'read {len(self.test_tokens)} tokens')

        self.train_vecs = self.vectorize(self.train_tokens)
        self.valid_vecs = self.vectorize(self.valid_tokens)
        self.test_vecs = self.vectorize(self.test_tokens)

    def tokenize(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = []
            for line in f:
                words = list()
                for word in line.split():
                    words.extend(self._word_to_allowable(word))

                tokens.extend(words)

        return tokens

    def vectorize(self, tokens):
        # Tokenize file content
        vecs = torch.zeros([len(tokens), self.w2c_model.vector_size],
                           dtype=torch.float32)
        for i, token in enumerate(tokens):
            vecs[i, :] = torch.tensor(self.w2c_model[token])

        return vecs

    def _word_to_allowable(self, word: str) -> List[str]:
        if word in self.w2c_model:
            return [word]
        # numbers
        if word.isnumeric():
            return [c for c in word]

        if len(word) <= 1:
            return []

        spl = -1
        for i, c in enumerate(word):
            if c not in self.w2c_model:
                spl = i
                break

        # Maybe there is an illegal character (/ , . etc) in this word
        # we should remove it and then recursively call on substrings
        if spl != -1:
            l = ''.join(list(word)[:spl])
            r = ''.join(list(word)[(spl+1):])
            return self._word_to_allowable(l) + self._word_to_allowable(r)

        return []

