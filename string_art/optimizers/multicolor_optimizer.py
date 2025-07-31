from copy import deepcopy
from typing import Union, Optional, Dict, Tuple
import os
from time import time

import numpy as np

from .base_optimizer import BaseOptimizer
from .preprocessing import preprocess_image
from string_art.utils import IterationCallback, save_image, save_fibers
from string_art.utils.visualizations import visualize_color_images, render_color_paths, print_with_colorbar, COLORBAR_PLACEHOLDER
from string_art.globals import LINES_TYPE, COLOR_IMAGES_TYPE, COLOR_TYPE, DebugFilenames
from string_art.configs.base_config import Config


class MulticolorOptimizer(BaseOptimizer):
    """
    Multicolor optimizer - reconstruct a colored image by dithering to get per-color 1-channel images and optimizing
    per color.
    """

    def preprocess(
            self,
            image: Union[str, np.ndarray],
            weights: Optional[Union[str, np.ndarray]] = None,
    ) -> Tuple[COLOR_IMAGES_TYPE, np.ndarray]:
        if isinstance(image, np.ndarray):
            assert image.ndim < 3, 'Using MulticolorOptimizer but received grayscale image.'

        aspect_ratio = self.canvas_cfg.size[0] / self.canvas_cfg.size[1]
        color_images, dithered = preprocess_image(
            image,
            resolution=self.preprocess_cfg.resolution,
            aspect_ratio=aspect_ratio,
            grayscale=False,
            colors=self.preprocess_cfg.colors,
            palette_type=self.preprocess_cfg.palette_type,
            n_colors=self.preprocess_cfg.n_colors,
            bg_color=self.canvas_cfg.bg_color,
        )
        self.canvas_cfg.resolution = dithered.shape[-2:]
        return color_images, dithered

    def optimize(
            self,
            image: Union[str, np.ndarray],
            weights: Optional[Union[str, np.ndarray]] = None,
            n_fibers: Optional[int] = None,
            callback_it: Optional[IterationCallback] = None,
            debug_path: Optional[str] = None,
            verbose: bool = False,
    ) -> Dict[COLOR_TYPE, LINES_TYPE]:
        # preprocess - get per-color images
        if verbose:
            start_time = time()
        color_images, dithered = self.preprocess(image)
        if verbose:
            preprocess_time = time()
        if n_fibers is None:
            n_fibers = self.optimizer_cfg.n_fibers
        if debug_path is not None:
            save_image(dithered, os.path.join(debug_path, DebugFilenames.DITHERED))
            visualize_color_images(color_images, os.path.join(debug_path, DebugFilenames.COLOR_IMAGES))

        # prepare BW optimizer
        from . import optimizer_factory_grayscale
        n_fibers_per_color = self._num_fibers_per_colors(color_images, n_total_fibers=n_fibers)
        bw_optimizer_cfg = deepcopy(self.optimizer_cfg)
        bw_optimizer_cfg.multicolor = False
        cfg = Config(self.canvas_cfg, self.preprocess_cfg, bw_optimizer_cfg)
        bw_optimizer = optimizer_factory_grayscale(cfg)
        if verbose:
            print('Number of lines per color:')
            max_width = max(len(str(color)) for color in n_fibers_per_color.keys())
            for i, (color, n_fibers) in enumerate(n_fibers_per_color.items()):
                print_with_colorbar(f'Color #{i} {COLORBAR_PLACEHOLDER} RGB: {str(color):<{max_width}} = {n_fibers}',
                                    color)

        # optimize per color
        paths = {}
        for i, color in enumerate(color_images.keys()):
            if verbose:
                print(f'Optimizing color #{i}:', end=' ')
            paths[color] = bw_optimizer.optimize(
                1 - color_images[color],  # preprocessing inverts image, so invert it here before,
                weights=weights,
                n_fibers=n_fibers_per_color[color],
                callback_it=callback_it,
            )

        if debug_path is not None:
            save_fibers(os.path.join(debug_path, DebugFilenames.PER_COLOR_FIBERS), paths)
            render_color_paths(paths, self.canvas_cfg, os.path.join(debug_path, DebugFilenames.COLOR_RESULTS))

        if verbose:
            optimize_time = time()
            print(f'Preprocessing time: {preprocess_time - start_time:.1f} sec')
            print(f'Optimization time: {optimize_time - preprocess_time:.1f} sec')
            print(f'Total time: {optimize_time - start_time:.1f} sec')
        return paths

    @staticmethod
    def _num_fibers_per_colors(color_images: COLOR_IMAGES_TYPE, n_total_fibers: int) -> Dict[COLOR_TYPE, int]:
        """
        Estimate number of fibers needed per color based on color histogram.
        :param color_images: dict of (h, w) image per color (scaled to [0, 1]).
        :param n_total_fibers:  total number of fibers.
        :return: dict of number of fibers per color.
        """
        n_fibers = {}
        total_sum = 0
        for color, color_image in color_images.items():
            n_fibers[color] = np.sum(color_image)
            total_sum += n_fibers[color]
        cum_fibers = 0
        for i, color in enumerate(n_fibers.keys()):
            if i < len(n_fibers) - 1:
                n_fibers[color] = int(n_fibers[color] * n_total_fibers / total_sum)
                cum_fibers += n_fibers[color]
            else:
                n_fibers[color] = n_total_fibers - cum_fibers
        return n_fibers

    def _optimize(
            self,
            image: Union[str, np.ndarray],
            weights: Optional[Union[str, np.ndarray]] = None,
            n_fibers: Optional[int] = None,
            callback_it: Optional[IterationCallback] = None,
            debug_path: Optional[str] = None,
    ):
        # no need to implement this because self.optimize() just calls BW optimizers.
        pass
