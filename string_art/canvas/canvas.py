from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING

import numpy as np
from skimage.draw import line_aa, line

from string_art.globals import LINES_TYPE, MAX_FIBER_VAL
from string_art.configs.base_config import CanvasConfig, Shape
from string_art.utils import save_image
if TYPE_CHECKING:  # use only for type checking to avoid circular imports
    from string_art.utils import IterationCallback


class Canvas:
    def __init__(self, config: CanvasConfig, for_optimization: bool = False):
        self.cfg = config
        self.canvas = np.zeros(self.cfg.resolution)
        # convert fiber width to pixel intensity by calculating what part of the pixel the fiber "covers"
        self._init_nails(self.cfg.shape)

        if for_optimization:  # check that fiber value is small enough for optimization
            scale = max(self.cfg.size[0] / self.cfg.resolution[0], self.cfg.size[1] / self.cfg.resolution[1])
            max_res = (int(MAX_FIBER_VAL * self.cfg.size[0] / self.cfg.fiber_width),
                       int(MAX_FIBER_VAL * self.cfg.size[1] / self.cfg.fiber_width))
            assert self.cfg.fiber_value <= MAX_FIBER_VAL, (
                f'Fiber width should be sufficiently smaller than pixel scale, but got (fiber_width = '
                f'{self.cfg.fiber_width} [mm]) and (pixel_scale = {scale:.2f} [mm/pixel]). For this pixel scale, use '
                f'fiber_width <= {MAX_FIBER_VAL * scale:.2f}. Alternatively, use resolution <= {max_res}.')

    # ----------------------------------------- Fiber drawing methods -----------------------------------------
    def add_fiber(self, nail1: int, nail2: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add fiber line between two nails to canvas.
        :return line_pixels: tuple of arrays with row and col line coordinates.
        """
        line_pixels, line_val = self.get_fiber(nail1, nail2, self.cfg.fiber_value)
        self.canvas[line_pixels] = self.canvas[line_pixels] + line_val
        return line_pixels

    def remove_fiber(self, nail1: int, nail2: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove fiber line between two nails to canvas.
        :return line_pixels: tuple of arrays with row and col line coordinates.
        """
        line_pixels, line_val = self.get_fiber(nail1, nail2, self.cfg.fiber_value)
        self.canvas[line_pixels] = self.canvas[line_pixels] - line_val
        return line_pixels

    def get_fiber(
            self,
            nail1: int,
            nail2: int,
            val: Optional[float] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Get representation of single line between two nails on the canvas.
        :param nail1: source nail of line.
        :param nail2: target nail of line.
        :param val: optional fiber intensity value (in [0, 1]), if None - will use self.cfg.fiber_value.
        :return line_pixels: tuple of arrays with row and col line coordinates.
        :return line_val: line_values (of same shape as row and col arrays).
        """
        if val is None:
            val = self.cfg.fiber_value
        r1, c1 = int(self.nails_rc[nail1, 0]), int(self.nails_rc[nail1, 1])
        r2, c2 = int(self.nails_rc[nail2, 0]), int(self.nails_rc[nail2, 1])
        if self.cfg.fiber_constant:
            line_r, line_c = line(r1, c1, r2, c2)
            line_val = val * np.ones_like(line_r)
        else:
            line_r, line_c, line_val = line_aa(r1, c1, r2, c2)
            line_val = val * line_val
        h, w = self.cfg.resolution
        line_valid = (line_r < h) & (line_c < w)
        line_pixels = (line_r[line_valid], line_c[line_valid])
        line_val = line_val[line_valid]
        return line_pixels, line_val

    # --------------------------------------- Error calculation methods ---------------------------------------
    def calc_error(
            self,
            image1: np.ndarray,
            image2: Optional[np.ndarray] = None,
            pixels: Optional[Tuple[np.ndarray, ...]] = None,
            weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate L2 error between a given image and current canvas.
        :param image1: image to calculate error against. (h, w)
        :param image2: if given, calculate error between two images and not canvas. (h, w)
        :param pixels: if given, evaluate error only over these pixels (tuple of per-dim index arrays).
        :param weights: if given, these are importance weights for error. (h, w)
        :return err: evaluated error.
        """
        if image2 is None:
            image2 = self.canvas
        if pixels is not None:
            mask = self.mask[pixels]
            image1 = image1[pixels]
            image2 = image2[pixels]
            if weights is not None:
                weights = weights[pixels]
        else:
            mask = self.mask
        if weights is None:
            weights = 1.
        err = weights * mask * (np.clip(image1, 0, 1) - np.clip(image2, 0, 1)) ** 2
        return float(np.sum(err))

    def simulate_line_improvement(
            self,
            image: np.ndarray,
            nail1: int,
            nail2: int,
            weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Simulate error improvement by adding a line.
        :param image: image to calculate error against. (h, w)
        :param nail1: source nail of line.
        :param nail2: target nail of line.
        :param weights: if given, these are importance weights for error. (h, w)
        :return: error improvement on the line.
        """
        line_pixels, line_val = self.get_fiber(nail1, nail2, self.cfg.fiber_value)
        weights = weights[line_pixels] if weights is not None else 1.
        err_before = np.sum((weights * (np.clip(self.canvas[line_pixels], 0, 1) - image[line_pixels])) ** 2)
        err_after = np.sum((weights * (np.clip(self.canvas[line_pixels] + line_val, 0, 1) - image[line_pixels])) ** 2)
        return err_before - err_after

    # ------------------------------------- Rendering and plotting methods -------------------------------------
    def render(
            self,
            lines: LINES_TYPE,
            callback: Optional[IterationCallback] = None,
            max_fibers: Optional[int] = None
    ):
        """
        Render a list of lines on the canvas.
        :param lines: a list of nails that form lines using a continuous fiber, or a list of pairs of nails.
        :param callback: optional callback to call each rendering iteration (each added fiber).
        :param max_fibers: max number of fibers to render.
        """
        if len(lines) == 0:
            return
        self.reset()
        if isinstance(lines[0], int):
            for i in range(1, len(lines)):
                self.add_fiber(lines[i - 1], lines[i])
                if callback is not None:
                    callback(i - 1, self)
                if max_fibers is not None and i >= max_fibers:
                    break
        elif isinstance(lines[0], Tuple):
            for i, (nail1, nail2) in enumerate(lines):
                self.add_fiber(nail1, nail2)
                if callback is not None:
                    callback(i, self)
                if max_fibers is not None and i + 1 >= max_fibers:
                    break

    def plot(self, show_nails: bool = True):
        import matplotlib.pyplot as plt
        plt.imshow(np.clip(self.canvas * self.mask, 0, 1), cmap='Grays')
        if show_nails:
            plt.scatter(self.nails_rc[:, 1], self.nails_rc[:, 0], c='r', s=0.2)
        plt.axis(False)
        plt.show()

    def save(self, path: str):
        save_image(self.get_image(), path)

    def reset(self):
        """
        Reset all lines from canvas.
        """
        self.canvas[:] = 0

    def get_image(self) -> np.ndarray:
        return 1 - np.clip(self.canvas * self.mask, 0, 1)

    # ---------------------------------------------- Init methods ----------------------------------------------
    def _init_nails(self, shape: Shape):
        """
        Distributes N nails evenly around the perimeter of the given shape.
        Assuming center of image is (0, 0), and normal angle convention (0 = horizontal).
        Initializes:
            nails_theta - angles of nails. (N)
            nails_rc - row-col positions of nails. (N, 2)
            mask - mask to erase pixels outside the shape. (h, w)
            valid_lines - list of lines (2-tuples of nail pairs) which are valid to connect
        """
        if shape == Shape.ELLIPSE:
            self._init_nails_circle()
        elif shape == Shape.RECTANGLE:
            self._init_nails_rectangle()
        else:
            raise NotImplementedError(f'canvas shape {shape} not implemented.')

        # add small permutation to avoid Moiré effect
        diff = np.mean(np.sqrt(np.sum((self.nails_rc - np.roll(self.nails_rc, -1, axis=0)) ** 2, axis=1)))
        var = 0.05 * diff
        self.nails_rc = self.nails_rc + np.random.randn(*self.nails_rc.shape) * (var ** 2)

    def _init_nails_circle(self):
        # init nail positions
        h, w = self.cfg.resolution
        r_x, r_y = w / 2, h / 2
        self.nails_theta = np.array([2 * np.pi * n / self.cfg.nails for n in range(self.cfg.nails)])[::-1]  # (N,)
        nails_x, nails_y = r_x * np.cos(self.nails_theta), r_y * np.sin(self.nails_theta)
        self.nails_rc = np.stack((-nails_y + h / 2, nails_x + w / 2), axis=1)  # (N, 2)

        # init mask
        y, x = np.ogrid[:h, :w]
        self.mask = ((x - w / 2) / r_x) ** 2 + ((y - h / 2) / r_y) ** 2 <= 1  # (h, w)

        # init valid lines
        self.valid_lines = []
        min_connected_distance = int(self.cfg.nails / 12)  # equivalent to 30 degrees per direction
        for nail1 in range(self.cfg.nails):
            # an ugly but efficient way to take all nail pairs with a minimal distance from nail1
            self.valid_lines += [(nail1, nail2 % self.cfg.nails) for nail2 in
                                 range(min(nail1 + min_connected_distance, self.cfg.nails),
                                       nail1 - min_connected_distance + 1 + self.cfg.nails) if nail2 < self.cfg.nails]

    def _init_nails_rectangle(self):
        h, w = self.cfg.resolution
        perimeter = 2 * (h + w)
        n_nails_h = int(h * self.cfg.nails / perimeter)
        n_nails_w = int(self.cfg.nails - n_nails_h * 2) // 2
        d_h = h / n_nails_h
        d_w = w / n_nails_w

        # add nails one by one, until reaching end of each edge
        nails_x, nails_y, nail_sides = [], [], []
        for i in range(n_nails_h):
            nails_x.append(w / 2)
            nails_y.append(h / 2 - i * d_h)
            nail_sides.append(0)
        for i in range(n_nails_w):
            nails_x.append(w / 2 - i * d_w)
            nails_y.append(-h / 2)
            nail_sides.append(1)
        for i in range(n_nails_h):
            nails_x.append(-w / 2)
            nails_y.append(-h / 2 + i * d_h)
            nail_sides.append(2)
        for i in range(n_nails_w):
            nails_x.append(-w / 2 + i * d_w)
            nails_y.append(h / 2)
            nail_sides.append(3)
        self.nails_theta = np.array([np.atan2(y, x) for x, y in zip(nails_x, nails_y)])  # (N,)
        self.nails_rc = np.stack((-np.array(nails_y) + h / 2, np.array(nails_x) + w / 2), axis=1)  # (N, 2)
        self.mask = np.ones(self.cfg.resolution)  # (h, w)

        # init valid lines
        self.valid_lines = []
        for nail1 in range(self.cfg.nails):
            # take only nail pairs which aren't on the same side of the rectangle
            self.valid_lines += [(nail1, nail2) for nail2 in range(nail1 + 1, self.cfg.nails) if
                                 nail_sides[nail1] != nail_sides[nail2]]
