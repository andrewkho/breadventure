#!/usr/bin/env python

from distutils.core import setup

setup(
    name='breadventure',
    version='1.0',
    description='LSTM Neural Network for bread recipes',
    author='Andrew Ho',
    author_email='andrew.kenneth.ho@gmail.com',
    url='https://github.com/andrewkho/breadventure',
    packages=['word2vec_lstm', 'word_language_model'],
    install_requires=['torch',
              'tensorflow',
              'tensorboard-pytorch',
              'numpy', 'gensim']
    )
