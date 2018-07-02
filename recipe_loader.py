import pickle
from typing import List

DEFAULT_DATA = 'data/recipes/recipes_complete.pkl'


class Recipe(object):
    pass


def load(pickle_file: str = DEFAULT_DATA) -> List[Recipe]:
    with open(pickle_file, 'rb') as f:
        recipes = pickle.load(f)

    return recipes
