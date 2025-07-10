from .base_config import *
from .bw_config import *
from .multicolor_config import *


def get_config(name: str) -> Config:
    name = name.upper()
    if name in globals():
        return globals()[name]
    raise ValueError(f'Configuration {name} not found, ensure it is defined correctly in one of the config files.')
