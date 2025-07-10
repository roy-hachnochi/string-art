from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING

import numpy as np

from .canvas import Canvas
from string_art.globals import COLOR_LINES_TYPE
from string_art.configs.base_config import CanvasConfig
if TYPE_CHECKING:  # use only for type checking to avoid circular imports
    from string_art.utils import IterationCallback


class MulticolorCanvas(Canvas):
    def __init__(self, config: CanvasConfig, for_optimization: bool = False):
        super(MulticolorCanvas, self).__init__(config, for_optimization)
        self.bg_color = self.cfg.bg_color
        self.opaque = not for_optimization
        self.reset()

    # ----------------------------------------- Fiber drawing methods -----------------------------------------
    def add_fiber(
            self,
            nail1: int,
            nail2: int,
            color: Optional[np.ndarray] = None,
            opaque: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add fiber line between two nails to canvas.
        :param color: array of size (3) indicating RGB color of added fiber (scaled to [0, 1]).
        :param opaque: if true, will add an opaque fiber on top of the current canvas, otherwise will mix by intensity.
        :return line_pixels: tuple of arrays with row and col line coordinates.
        """
        if color is None:
            color = np.array([1., 1., 1.])
        fiber_value = 1. if opaque else self.cfg.fiber_value
        line_pixels, line_val = self.get_fiber(nail1, nail2, fiber_value)
        self.canvas[:, *line_pixels] = line_val * color[:, np.newaxis] + (1 - line_val) * self.canvas[:, *line_pixels]
        return line_pixels

    def remove_fiber(
            self,
            nail1: int,
            nail2: int,
            color: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove fiber line between two nails to canvas.
        :param color: array of size (3) indicating RGB color of added fiber (scaled to [0, 1]).
        :return line_pixels: tuple of arrays with row and col line coordinates.
        """
        if color is None:
            color = np.array([1., 1., 1.])
        line_pixels, line_val = self.get_fiber(nail1, nail2, self.cfg.fiber_value)
        self.canvas[:, *line_pixels] = self.canvas[:, *line_pixels] - line_val * color[:, np.newaxis]
        return line_pixels

    # --------------------------------------- Error calculation methods ---------------------------------------
    def calc_error(
            self,
            image1: np.ndarray,
            image2: Optional[np.ndarray] = None,
            pixels: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate L2 error between a given image and current canvas.
        :param image1: image to calculate error against. (3, h, w)
        :param image2: if given, calculate error between two images and not canvas. (3, h, w)
        :param pixels: if given, evaluate error only over these pixels.
        :param weights: if given, these are importance weights for error. (h, w)
        :return err: evaluated error.
        """
        pixels_3d = (np.arange(3)[:, np.newaxis], pixels[0], pixels[1])  # add color channel
        return super(MulticolorCanvas, self).calc_error(image1, image2, pixels_3d, weights)

    def simulate_line_improvement(
            self,
            image: np.ndarray,
            nail1: int,
            nail2: int,
            color: Optional[np.ndarray] = None,
            weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Simulate error improvement by adding a line.
        :param image: image to calculate error against.
        :param nail1: source nail of line.
        :param nail2: target nail of line.
        :param color: array of size (3) indicating RGB color of added fiber (scaled to [0, 1]).
        :param weights: if given, these are importance weights for error. (h, w)
        :return: error improvement on the line.
        """
        if color is None:
            color = np.array([1., 1., 1.])
        line_pixels, line_val = self.get_fiber(nail1, nail2, self.cfg.fiber_value)
        weights = weights[line_pixels] if weights is not None else 1.
        err_before = np.sum((weights * (np.clip(self.canvas[:, *line_pixels], 0, 1) - image[:, *line_pixels])) ** 2)
        err_after = np.sum((weights * (np.clip(
                    line_val * color[:, np.newaxis] + (1 - line_val) * self.canvas[:, *line_pixels], 0, 1) -
                    image[:, *line_pixels])) ** 2)
        return err_before - err_after

    # ------------------------------------- Rendering and plotting methods -------------------------------------
    def render(
            self,
            lines: COLOR_LINES_TYPE,
            callback: Optional[IterationCallback] = None,
            max_fibers: Optional[int] = None,
            opaque: Optional[bool] = None,
    ):
        """
        Render a list of lines on the canvas.
        :param lines: list of tuples representing a colored path, each tuple is (color, path segment), each color is a
            3-tuple representing the color in RGB uint8, and each path segment is a list representing the lines in the
            path of this color (see COLOR_LINES_TYPE).
        :param callback: optional callback to call each rendering iteration (each added fiber).
        :param max_fibers: max number of fibers to render.
        :param opaque: if true, will add render opaque fibers on top of each other, otherwise will blend them
            (default: self.opaque).
        """
        if len(lines) == 0:
            return
        if opaque is None:
            opaque = self.opaque
        self.reset()
        it = 0
        cur_nail_per_color = {}  # keeps track of current nail, for continuous path
        for color, path in lines:
            color_np = np.array(color, dtype=np.float32) / 255.
            if path and isinstance(path[0], int):  # continuous
                path_iter = iter(path)
                if color not in cur_nail_per_color:  # starting path for new color
                    cur_nail_per_color[color] = next(path_iter)
                nail1 = cur_nail_per_color[color]
                for nail2 in path_iter:
                    self.add_fiber(nail1, nail2, color_np, opaque)
                    nail1 = nail2
                    if callback is not None:
                        callback(it, self)
                    it += 1
                cur_nail_per_color[color] = nail1
            elif path and isinstance(path[0], Tuple):  # non-continuous
                for nail1, nail2 in path:
                    self.add_fiber(nail1, nail2, color_np, opaque)
                    if callback is not None:
                        callback(it, self)
                    it += 1
            if max_fibers is not None and it >= max_fibers:
                break

    def set_bg_color(self, color: Optional[Tuple[int, int, int]] = None):
        """
        Set new background color for canvas to be used from now on.
        :param color: 3-tuple of new canvas background color (specified in RGB in range [0, 255]),
            if not given - set to config.bg_color default.
        """
        if color is not None:
            self.bg_color = color
        else:
            self.bg_color = self.cfg.bg_color
        self.reset()

    def plot(self, show_nails: bool = True):
        import matplotlib.pyplot as plt
        image = np.ones((3,) + self.cfg.resolution)
        image = image * (1 - self.mask) + self.canvas * self.mask
        plt.imshow(np.clip(image, 0, 1).transpose((1, 2, 0)))
        if show_nails:
            plt.scatter(self.nails_rc[:, 1], self.nails_rc[:, 0], c='r', s=0.2)
        plt.axis(False)
        plt.show()

    def get_image(self) -> np.ndarray:
        image = np.ones((3,) + self.cfg.resolution)
        image = image * (1 - self.mask) + self.canvas * self.mask
        return np.clip(image, 0, 1)

    def reset(self):
        color_array = np.array(self.bg_color, dtype=np.float64).reshape(3, 1, 1) / 255
        self.canvas = np.ones((3,) + self.cfg.resolution, dtype=color_array.dtype) * color_array
