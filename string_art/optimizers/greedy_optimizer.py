from typing import Union, Tuple, Optional, Set
from tqdm import tqdm
import random

import numpy as np

from .base_optimizer import BaseOptimizer
from string_art.utils import IterationCallback
from string_art.globals import LINES_TYPE


class GreedyOptimizer(BaseOptimizer):
    """
    Greedy optimizer - attempt to reconstruct the image by adding lines one by one, minimizing the reconstruction error.
    self.optimizer_cfg.continuous - enforce forming a continuous path between nails during optimization, otherwise
    chooses best line from all possibilities
    """

    def _optimize(
            self,
            image: np.ndarray,
            weights: Optional[np.ndarray] = None,
            n_fibers: Optional[int] = None,
            callback_it: Optional[IterationCallback] = None,
            debug_path: Optional[str] = None,
    ) -> LINES_TYPE:
        if n_fibers is None:
            n_fibers = self.optimizer_cfg.n_fibers

        # greedily find next best line
        path = []
        used_lines = set()
        curr_nail = 0
        prev_err = float('inf')
        for it in tqdm(range(n_fibers), desc='GreedyOptimizer'):
            # find next best line
            if self.optimizer_cfg.continuous:  # continue from last nail
                next_nail, _ = self._find_best_line_from_nail(image, curr_nail, used_lines, weights=weights)
                best_line = (curr_nail, next_nail)
                curr_nail = next_nail
            else:  # don't enforce path continuity (best not to use this option, very inefficient)
                best_line = self._find_best_line(image, used_lines, weights=weights)

            # add line and break if done
            self.canvas.add_fiber(*best_line)
            err = self.canvas.calc_error(image, weights=weights)
            if err >= prev_err:
                break
            if callback_it is not None:
                callback_it(it, self.canvas)
            path.append(best_line)
            used_lines.add(best_line)
            if err < self.optimizer_cfg.error_threshold:
                break
            prev_err = err

        if self.optimizer_cfg.continuous:
            path = [path[0][0]] + [nail2 for nail1, nail2 in path]
        return path

    def _find_best_line(
            self,
            image: np.ndarray,
            used_lines: Set[Tuple[int, int]],
            weights: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        """
        Find best line from any two nails, that minimizes the image error.
        :param image: image to attempt to reconstruct using the line. (h, w)
        :param used_lines: lines which were already added to the path, to skip them.
        :param weights: if given, these are importance weights for error. (h, w)
        :return best: next optimal line represented by two nails.
        """
        max_err_improvement = float('-inf')
        best = (0, 0)
        for nail1 in range(self.canvas_cfg.nails):
            nail2, err_improvement = self._find_best_line_from_nail(image, nail1, used_lines, weights=weights)
            if err_improvement > max_err_improvement:
                max_err_improvement = err_improvement
                best = (nail1, nail2)
        return best

    def _find_best_line_from_nail(
            self,
            image: np.ndarray,
            nail: int,
            used_lines: Set[Tuple[int, int]],
            weights: Optional[np.ndarray] = None,
    ) -> Tuple[int, float]:
        """
        Find best line from a given nail, that minimizes the image error.
        :param image: image to attempt to reconstruct using the line. (h, w)
        :param nail: start nail of the new line.
        :param used_lines: lines which were already added to the path, to skip them.
        :param weights: if given, these are importance weights for error. (h, w)
        :return next_nail: next optimal nail to make a line from current nail to.
        :return max_err_improvement: best achieved error improvement using new line.
        """
        max_err_improvement = float('-inf')
        best = (0, 0)
        valid_lines = [line for line in self.canvas.valid_lines if nail in line]
        if self.optimizer_cfg.n_random_nails is not None and self.optimizer_cfg.n_random_nails > 0:
            n = min(len(valid_lines), self.optimizer_cfg.n_random_nails)
            valid_lines = random.sample(valid_lines, n)
        for line in valid_lines:
            # skip used lines
            if (line[0], line[1]) in used_lines or (line[1], line[0]) in used_lines:
                continue

            # evaluate error improvement and update if better
            err_improvement = self.canvas.simulate_line_improvement(image, line[0], line[1], weights=weights)
            if err_improvement > max_err_improvement:
                max_err_improvement = err_improvement
                best = line
        next_nail = best[0] if best[0] != nail else best[1]  # the nail in best line which isn't the current nail
        return next_nail, max_err_improvement
