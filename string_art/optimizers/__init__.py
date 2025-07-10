from .base_optimizer import BaseOptimizer
from .multicolor_optimizer import MulticolorOptimizer
from .greedy_optimizer import GreedyOptimizer
from .linear_optimizer import LinearModelOptimizer
from .binary_linear_optimizer import BinaryLinearOptimizer
from .multicolor_binary_linear_optimizer import MulticolorBinaryLinearOptimizer
from .fibers_path import FibersPath

from string_art.configs.base_config import Config, OptimizerType


def optimizer_factory(config: Config) -> BaseOptimizer:
    if config.optimizer.multicolor:
        if config.optimizer.type == OptimizerType.MULTICOLOR_BINARY_LINEAR:
            return MulticolorBinaryLinearOptimizer(config)
        else:
            return MulticolorOptimizer(config)
    else:
        return optimizer_factory_grayscale(config)


def optimizer_factory_grayscale(config: Config) -> BaseOptimizer:
    if config.optimizer.type == OptimizerType.GREEDY:
        return GreedyOptimizer(config)
    elif config.optimizer.type == OptimizerType.LEAST_SQUARES:
        return LinearModelOptimizer(config)
    elif config.optimizer.type == OptimizerType.BINARY_LINEAR:
        return BinaryLinearOptimizer(config)
    else:
        raise ValueError(f'Optimizer type {config.optimizer.type} not implemented, see OptimizerType for valid types.')
