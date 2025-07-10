import pickle

from .image import *
from .callbacks import *
from .argparser import *

def save_fibers(path: str, fibers):
    with open(path, 'wb') as f:
        pickle.dump(fibers, f)
