import numpy as np

import nltk
import logging
from typing import List

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PATH_TO_NYC = 'data/nyc-format'
RECIPE_TEXTS = [f'bread{c}.txt' for c in ('', '2', '3')]
OUT_PREPEND = 'nyc_'

SEED=123


def run():
    logger.info('Processing recipes in %s', PATH_TO_NYC)
    raw_lines = []
    for filename in RECIPE_TEXTS:
        with open(f'{PATH_TO_NYC}/{filename}', 'r', errors='ignore') as f:
            raw_lines.extend(f.readlines())
    logger.info('Splitting into recipes')
    recipes = split_into_recipes(raw_lines)
    # logger.info('  passing through nltk')
    # nltkd = []
    # for r in recipes:
    #     nltkd.append(prelim_clean(r))

    logger.info('  Split into train/valid/test')
    test_size = int(0.1*len(recipes))
    valid_size = test_size
    train_size = len(recipes) - test_size - valid_size
    np.random.seed(SEED)
    np.random.shuffle(recipes)

    train = recipes[:train_size]
    valid = recipes[train_size:train_size+valid_size]
    test = recipes[train_size+valid_size:]

    fname_out = f'{PATH_TO_NYC}/{OUT_PREPEND}train.txt'
    logger.info('  Writing to %s', fname_out)
    with open(fname_out, 'w') as f:
        f.writelines('\n\n'.join(train))

    fname_out = f'{PATH_TO_NYC}/{OUT_PREPEND}valid.txt'
    logger.info('  Writing to %s', fname_out)
    with open(fname_out, 'w') as f:
        f.writelines('\n\n'.join(valid))

    fname_out = f'{PATH_TO_NYC}/{OUT_PREPEND}test.txt'
    logger.info('  Writing to %s', fname_out)
    with open(fname_out, 'w') as f:
        f.writelines('\n\n'.join(test))

    logger.info('  Done!')


def split_into_recipes(lines: List[str]) -> List[str]:
    """

    :param lines:
    :return:
    """
    START_LINE = '@@@@@ Now You\'re Cooking!'.lower()
    END_LINE = '** Exported from Now You\'re Cooking!'.lower()

    recipes = []
    r = []
    in_recipe = False
    for line in lines:
        # Look for start of recipe
        if not in_recipe:
            if line.lower().find(START_LINE) >= 0:
                in_recipe = True
            continue

        # Look for end of recipe to stop
        if line.lower().find(END_LINE) >= 0:
            in_recipe = False
            if len(r) > 0:
                r = [prelim_clean(_) for _ in r]
                recipes.append('\n'.join(r))
            r = []
            continue

        # Skip nutrilink lines
        if line.lower().find('NYC Nutrilink'.lower()) >= 0:
            continue

        r.append(line)

    return recipes


def remove_nyc(lines: List[str]) -> List[str]:
    return [_ for _ in lines if _.find("Now You're Cooking!") >= 0]


def prelim_clean(text: str) -> str:
    return ' '.join(nltk.tokenize.word_tokenize(text)).lower().strip()


if __name__ == '__main__':
    run()
