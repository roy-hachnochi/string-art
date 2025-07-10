from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
from copy import deepcopy
from time import time
import os

import numpy as np

from string_art.utils import IterationCallback, save_fibers
from string_art.configs.base_config import Config
from string_art.canvas import Canvas
from string_art.optimizers.preprocessing import preprocess_image
from string_art.globals import LINES_TYPE, DebugFilenames


class BaseOptimizer(ABC):
    """
    Abstract base class for optimizing a canvas given an image. Inheritors need to implement self.optimize(...).
    """

    def __init__(self, config: Config):
        self.canvas_cfg = deepcopy(config.canvas)
        self.optimizer_cfg = deepcopy(config.optimizer)
        self.preprocess_cfg = deepcopy(config.preprocess)
        self.canvas = None  # initialized in preprocess

    def preprocess(
            self,
            image: Union[str, np.ndarray],
            weights: Optional[Union[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess an image and initialize optimizer based on inputs.
        :param image: image to optimize (scaled to [0, 1]), if given string - will load the image. (h, w) or (c, h, w)
        :param weights: optimization importance weights (scaled to [0, 1]), 0 (black) is a high weight and 1 (white) is
            a low weight, if given string - will load the image. (h, w)
        :return: preprocessed image and weights.
        """
        aspect_ratio = self.canvas_cfg.size[0] / self.canvas_cfg.size[1]
        image = preprocess_image(image, resolution=self.preprocess_cfg.resolution, aspect_ratio=aspect_ratio, grayscale=True)
        self.canvas_cfg.resolution = image.shape[-2:]
        if weights is not None:
            weights = preprocess_image(weights, resolution=self.canvas_cfg.resolution, grayscale=True)
        self.canvas = Canvas(self.canvas_cfg, for_optimization=True)
        self._init(weights)
        return image, weights

    def _init(self, weights: Optional[np.ndarray] = None):
        """
        Class inheritors may implement this to initialize specific instance variables, called at end of preprocess.
        """
        pass

    def optimize(
            self,
            image: Union[str, np.ndarray],
            weights: Optional[Union[str, np.ndarray]] = None,
            n_fibers: Optional[int] = None,
            callback_it: Optional[IterationCallback] = None,
            debug_path: Optional[str] = None,
            verbose: bool = False,
    ) -> LINES_TYPE:
        """
        Optimize an image - find a set of fibers/chords/lines such that rendering them will reconstruct the image.
        :param image: image to optimize (scaled to [0, 1]), if given string - will load the image. (h, w) or (c, h, w)
        :param weights: optimization importance weights (scaled to [0, 1]), 0 (black) is a high weight and 1 (white) is
            a low weight, if given string - will load the image. (h, w)
        :param n_fibers: optional maximal number of fibers allowed for result.
        :param callback_it: optional callback called each optimization iteration (e.g., save/plot the current result).
        :param debug_path: optional debug path to save debug artifacts.
        :param verbose: add logging messages during optimization.
        :return: a path of nails on the canvas to pass fibers through, or a set of lines indicated by nail-pairs.
        """
        # preprocess
        if verbose:
            start_time = time()
        image, weights = self.preprocess(image, weights)
        self.canvas.reset()
        if verbose:
            preprocess_time = time()

       # optimize
        path = self._optimize(image, weights, n_fibers, callback_it, debug_path)
        if verbose:
            optimize_time = time()
            print(f'Preprocessing time: {preprocess_time - start_time:.1f} sec')
            print(f'Optimization time: {optimize_time - preprocess_time:.1f} sec')
            print(f'Total time: {optimize_time - start_time:.1f} sec')
        if debug_path:
            save_fibers(os.path.join(debug_path, DebugFilenames.FIBERS), path)

        return path

    @abstractmethod
    def _optimize(
            self,
            image: np.ndarray,
            weights: Optional[np.ndarray] = None,
            n_fibers: Optional[int] = None,
            callback_it: Optional[IterationCallback] = None,
            debug_path: Optional[str] = None,
    ) -> LINES_TYPE:
        """
        Optimize an image - find a set of fibers/chords/lines such that rendering them will reconstruct the image.
        :param image: image to optimize (scaled to [0, 1]). (h, w) or (c, h, w)
        :param weights: optimization importance weights (scaled to [0, 1]), 0 (black) is a high weight and 1 (white) is
            a low weight. (h, w)
        :param n_fibers: optional maximal number of fibers allowed for result.
        :param callback_it: optional callback called each optimization iteration (e.g., save/plot the current result).
        :param debug_path: optional debug path to save debug artifacts.
        :return: a path of nails on the canvas to pass fibers through, or a set of lines indicated by nail-pairs.
        """
        raise NotImplementedError('BaseOptimizer inheritors should implement _optimize method.')
