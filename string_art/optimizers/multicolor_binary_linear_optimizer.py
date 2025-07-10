from typing import Union, Optional, Tuple, List
import os
from time import time

from tqdm import tqdm
import numpy as np

from .linear_optimizer import LinearModelOptimizer
from .preprocessing import preprocess_image
from string_art.utils import save_fibers
from string_art.utils import IterationCallback, save_image
from string_art.utils.visualizations import visualize_color_images, print_with_colorbar, COLORBAR_PLACEHOLDER
from string_art.globals import COLOR_IMAGES_TYPE, DebugFilenames, COLOR_LINES_TYPE, SEQUENCE_LENGTH, MIN_SEQUENCE
from string_art.canvas import MulticolorCanvas
from string_art.configs import Config


class MulticolorBinaryLinearOptimizer(LinearModelOptimizer):
    """
    Multicolor optimizer - same as BinaryLinearOptimizer, but for mutilcolor image by modelling the problem as:
    min. ||AxC-b|| s.t. x in {0,1}^N. Where:
    - b is the flattened colored target image (n_pixels, 3).
    - x is a binary vector stating which lines are to be added and in what color (n_lines, n_colors).
    - A is the transformation that sums all lines to an image (n_pixels, n_lines).
    - C is the dictionary of possible colors (n_colors, 3).
    """

    def __init__(self, config: Config):
        super(MulticolorBinaryLinearOptimizer, self).__init__(config)
        assert config.optimizer.continuous, 'MulticolorBinaryLinearOptimizer doesn\'t support non-continuous path.'
        self.bg_color = 0.5 * np.ones(3)
        # TODO: centering colors around gray so that black and white colors will also be treated, this implicitly
        #  assumes that gray colors aren't present because they will be disregarded as BG.
        #  Instead, we can embed all colors to a 4D sphere so that they will ALL be treated the same.

    def preprocess(
            self,
            image: Union[str, np.ndarray],
            weights: Optional[Union[str, np.ndarray]] = None,
    ) -> Tuple[COLOR_IMAGES_TYPE, np.ndarray]:
        if isinstance(image, np.ndarray):
            assert image.ndim < 3, 'Using MulticolorBinaryLinearOptimizer but received grayscale image.'
        aspect_ratio = self.canvas_cfg.size[0] / self.canvas_cfg.size[1]
        color_images, dithered = preprocess_image(
            image,
            resolution=self.preprocess_cfg.resolution,
            aspect_ratio=aspect_ratio,
            grayscale=False,
            colors=self.preprocess_cfg.colors,
            rgbcmykw=self.preprocess_cfg.rgbcmykw,
            n_colors=self.preprocess_cfg.n_colors,
            bg_color=self.canvas_cfg.bg_color,
        )
        self.canvas_cfg.resolution = dithered.shape[-2:]
        if weights is not None:
            weights = preprocess_image(weights, resolution=self.canvas_cfg.resolution, grayscale=True)
        self.canvas = MulticolorCanvas(self.canvas_cfg, for_optimization=True)
        self._init(weights)
        return color_images, dithered

    def optimize(
            self,
            image: Union[str, np.ndarray],
            weights: Optional[Union[str, np.ndarray]] = None,
            n_fibers: Optional[int] = None,
            callback_it: Optional[IterationCallback] = None,
            debug_path: Optional[str] = None,
            verbose: bool = False,
    ) -> COLOR_LINES_TYPE:
        # preprocess - get dithered image and fiber colors
        if verbose:
            start_time = time()
        color_images, dithered = self.preprocess(image, weights)
        colors = list(color_images.keys())
        if verbose:
            preprocess_time = time()
            print('Using string colors:')
            max_width = max(len(str(color)) for color in colors)
            for i, color in enumerate(colors):
                print_with_colorbar(f'Color #{i} {COLORBAR_PLACEHOLDER} RGB: {str(color):<{max_width}}', color)
        if debug_path is not None:
            save_image(dithered, os.path.join(debug_path, DebugFilenames.DITHERED))
            visualize_color_images(color_images, os.path.join(debug_path, DebugFilenames.COLOR_IMAGES))

        # setup
        self._setup(dithered, colors)
        if n_fibers is None:
            n_fibers = self.optimizer_cfg.n_fibers

        # optimize
        best_err = (0, float(self.err))
        for it in tqdm(range(n_fibers), desc='MulticolorBinaryLinearOptimizer'):
            k, c, err = self._find_best_line()
            self._update(k, c, err)
            if callback_it is not None:
                self.canvas.add_fiber(*self.nail_pairs[k], 1 - np.exp(self.C[c]) + self.bg_color,
                                      opaque=False)  # multiplicative model
                callback_it(it, self.canvas)
            best_err = min(best_err, (it + 1, err), key=lambda x: x[1])
            if self.err < self.optimizer_cfg.error_threshold:
                break

        # trim lines at best error
        lines = []
        it = 0
        for color, path in self.lines:
            lines.append((color, []))
            for nail in path:
                lines[-1][1].append(nail)
                it += 1
                if it >= best_err[0]:
                    break
            if it >= best_err[0]:
                break

        # reverse lines because they're ordered by decreasing importance
        lines = [(color, path[::-1]) for color, path in reversed(lines)]
        if debug_path is not None:
            np.save(os.path.join(debug_path, DebugFilenames.LINES_VECTOR), self.x)
            save_fibers(os.path.join(debug_path, DebugFilenames.PER_COLOR_FIBERS), lines)
        if verbose:
            optimize_time = time()
            print(f'Preprocessing time: {preprocess_time - start_time:.1f} sec')
            print(f'Optimization time: {optimize_time - preprocess_time:.1f} sec')
            print(f'Total time: {optimize_time - start_time:.1f} sec')
        return lines

    def _find_best_line(self) -> Tuple[int, int, float]:
        """
        Find best line to add for given color, that minimizes the error.
        :return k: next line index to add.
        :return c: color index of next line.
        :return err: updated error after adding line.
        """
        # check if allowing new colors or enforcing same color as previous
        if len(self.lines) > 0 and (len(self.lines[-1][1]) % SEQUENCE_LENGTH != 0 or len(
                self.lines[-1][1]) < MIN_SEQUENCE):
            enforce_color = list(self.cur_nails.keys()).index(self.lines[-1][0])
        else:
            enforce_color = None

        # get valid lines per color
        valid_lines_l = []
        for color_i, nail in enumerate(self.cur_nails.values()):
            if enforce_color is None or color_i == enforce_color:
                if nail is not None:
                    lines = np.array(self.nail_to_lines[nail])
                else:
                    lines = np.arange(self.x.shape[0])
                lines = lines[np.all(self.x[lines, :] == 0, axis=1)]
                if self.optimizer_cfg.n_random_nails is not None and self.optimizer_cfg.n_random_nails > 0:
                    n = min(len(lines), self.optimizer_cfg.n_random_nails)
                    lines = np.random.choice(lines, n, replace=False)
                valid_lines_l.append(lines)
        valid_lines = np.zeros((max(len(lines) for lines in valid_lines_l), len(valid_lines_l)), dtype=np.int32)
        for i, lines in enumerate(valid_lines_l):
            valid_lines[:len(lines), i] = lines  # (n_lines, n_colors)

        # precalculate norm and inner product
        A = self.A[:, valid_lines.ravel()]  # (n_pixels, n_lines x n_colors)
        C = self.C[[enforce_color], :] if enforce_color is not None else self.C  # (n_colors, 3)
        norms = self.norms[valid_lines, enforce_color] if enforce_color is not None else self.norms[
            valid_lines, np.arange(valid_lines.shape[1])]  # (n_lines, n_colors)
        inner_products = (self.residual.T @ A).reshape(3, *valid_lines.shape).transpose(1, 2, 0)  # (n_lines, n_colors, 3)
        inner_products = np.sum(inner_products * C[np.newaxis, :, :], axis=2)  # (n_lines, n_colors)

        # find best line to add
        add_errs = self.err + norms + 2 * inner_products  # (n_lines, n_colors)
        best_idx, next_color_idx = np.unravel_index(np.argmin(add_errs), add_errs.shape)
        next_idx = valid_lines[best_idx, next_color_idx]
        next_err = float(np.min(add_errs))
        if enforce_color is not None:
            next_color_idx = enforce_color
        return int(next_idx), int(next_color_idx), next_err

    def _update(self, k: int, color_i: int, err: float):
        """
        Add or remove a line, and update state.
        :param k: line index to add.
        :param color_i: color index of next line.
        :param err: error achieved by this update.
        """
        color = list(self.cur_nails.keys())[color_i]
        self.x[k, color_i] = 1
        nonzero = self.A[:, k].nonzero()[0]
        self.residual[nonzero, :] += self.A[nonzero, k].data[:, np.newaxis] * self.C[color_i, :]
        self.n_fibers += 1
        self.err = err

        cur_nail = self.cur_nails[color]
        nail_pair = self.nail_pairs[k]
        next_nail = nail_pair[0] if nail_pair[1] == cur_nail else nail_pair[1]
        if cur_nail is None:  # new color - add both nails as first line
            self.lines.append((color, [*nail_pair]))
        elif len(self.lines) > 0 and self.lines[-1][0] == color:  # previous line was from this color - add to it's path
            self.lines[-1][1].append(next_nail)
        else:  # switching colors - add new color path
            self.lines.append((color, [next_nail]))
        self.cur_nails[color] = next_nail

    def _setup(self, image: np.ndarray, colors: List[Tuple[int, int, int]]):
        """
        Initializes residuals for efficient error calculation:
            residual: (n_pixels, 3) array holding the current residual (r_k = A*x_k*C - b).
            err: current error (err_k = ||r_k||^2).
            x: (n_lines, n_colors) array with colors of each line used in solution.
            C: (n_colors, 3) array with available colors.
            norms: (n_lines, n_colors) norms of each column of A of each color.
            lines: lines used in current solution.
            n_fibers: number of lines used in current solution (sparsity of x).
            cur_nails: current nail per color.
        """
        b = image.reshape(-1, image.shape[1] * image.shape[2]).transpose() - self.bg_color[np.newaxis, :]
        b = np.log(1 - b)  # multiplicative model
        self.residual = -b
        self.err = np.sum(self.residual ** 2)
        self.x = np.zeros((len(self.nail_pairs), len(colors)))
        self.C = np.array(colors, dtype=np.float32) / 255. - self.bg_color[np.newaxis, :]
        self.C = np.log(1 - self.C)  # multiplicative model
        self.norms = np.outer(np.array(self.A.power(2).sum(axis=0)), np.sum(self.C ** 2, axis=1))
        self.lines = []
        self.n_fibers = 0
        self.cur_nails = {color: None for color in colors}

    def _optimize(
            self,
            image: np.ndarray,
            weights: Optional[np.ndarray] = None,
            n_fibers: Optional[int] = None,
            callback_it: Optional[IterationCallback] = None,
            debug_path: Optional[str] = None,
    ) -> COLOR_LINES_TYPE:
        # no need to implement this because self.optimize() implements the optimization
        pass
