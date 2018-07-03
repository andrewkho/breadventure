import re
import numpy as np
import nltk

import recipe_loader
from recipe_loader import Recipe


TEXTDIR = 'data/recipes/text/'
DIRECTIONS = 'data/recipes/directions/'
INGREDIENTS = 'data/recipes/ingredients/'
DESCRIPTIONS = 'data/recipes/descriptions/'


def split_recipes(seed = 123):
    recipes = recipe_loader.load()

    test_size = int(0.1*len(recipes))
    valid_size = test_size
    train_size = len(recipes) - test_size - valid_size

    np.random.seed(seed)
    np.random.shuffle(recipes)
    train = recipes[:train_size]
    valid = recipes[train_size:train_size+valid_size]
    test = recipes[train_size+valid_size:]

    print(f'{len(train)} training, {len(valid)} validation, {len(test)} test')

    # Write text
    def write_raw(fname, _rec):
        with open(TEXTDIR + fname, 'w') as f:
            f.write('\n'.join([clean(_) for _ in _rec]).lower())
    write_raw('train.txt', train)
    write_raw('valid.txt', valid)
    write_raw('test.txt', test)

    def write_ing(fname, _rec):
        with open(INGREDIENTS + fname, 'w') as f:
            f.write('\n'.join([_.text['ingreds'] for _ in _rec]).lower())
    write_ing('train.txt', train)
    write_ing('valid.txt', valid)
    write_ing('test.txt', test)

    def write_desc(fname, _rec):
        with open(DESCRIPTIONS + fname, 'w') as f:
            f.write('\n'.join([_.text['description'] for _ in _rec]).lower())
    write_desc('train.txt', train)
    write_desc('valid.txt', valid)
    write_desc('test.txt', test)

    def write_dir(fname, _rec):
        with open(DIRECTIONS + fname, 'w') as f:
            f.write('\n'.join([_.text['directions'] for _ in _rec]).lower())
    write_dir('train.txt', train)
    write_dir('valid.txt', valid)
    write_dir('test.txt', test)


def clean(recipe):
    """ Clean a recipe and return the raw text
    """
    # Start with title
    raw = _preliminary_clean(
        _split_asterisk(recipe.title)) + '\n\n'
    raw += _preliminary_clean(
        _split_asterisk(recipe.text['description'])) + '\n\n'
    raw += _split_ingreds(
        _split_asterisk(recipe.text['ingreds'])) + '\n\n'
    raw += _split_directions(
        _split_asterisk(recipe.text['directions'])) + '\n\n'
    if 'tips' in recipe.text:
        raw += _split_tips(
            _split_asterisk(recipe.text['tips'])) + '\n\n'

    return raw


def _split_asterisk(s):
    return re.sub(r'\*', r' * ', s)


def _preliminary_clean(s):
    return ' '.join(nltk.tokenize.word_tokenize(s)).lower().strip()


def _split_ingreds(s):
    return '\n'.join([' '.join([split_trailing_g(_)
                                for _ in nltk.tokenize.word_tokenize(_)])
                      for _ in s.split('\n')]).lower().strip()


def _split_tips(s):
    return '\n'.join([' '.join(nltk.tokenize.word_tokenize(_))
                      for _ in s.strip().split('\n')])


def _split_directions(s):
    return ' '.join(nltk.tokenize.word_tokenize(s.strip())[1:]).lower().strip()


def split_trailing_g(s):
    return re.sub(r'([0-9].*)g$', r'\1 g', s)


if __name__ == '__main__':
    split_recipes()
