from typing import Tuple, Optional
from tqdm import tqdm
import os
from copy import copy
import random

import numpy as np

from .linear_optimizer import LinearModelOptimizer
from string_art.utils import IterationCallback
from string_art.globals import LINES_TYPE, DebugFilenames


class BinaryLinearOptimizer(LinearModelOptimizer):
    """
    Binary linear model optimizer - Model the string-art problem as a sparse linear optimization problem:
    min. ||Ax-b|| s.t. x in {0,1}^N. Where:
    - b is the flattened target image.
    - x is a binary vector stating which lines are to be added.
    - A is the transformation that sums all lines to an image.

    Explicitly enforce sparsity and binarity by solving greedily using sparse matrices and updating residuals.
    """

    def _optimize(
            self,
            image: np.ndarray,
            weights: Optional[np.ndarray] = None,
            n_fibers: Optional[int] = None,
            callback_it: Optional[IterationCallback] = None,
            debug_path: Optional[str] = None,
    ) -> LINES_TYPE:
        # setup
        b = image.flatten()
        self._setup(b)
        if n_fibers is None:
            n_fibers = self.optimizer_cfg.n_fibers

        # optimize
        progress = tqdm(total=n_fibers, desc='BinaryLinearOptimizer')
        it = 0
        nail = None
        best_err = (0, float(self.err))
        while self.n_fibers < n_fibers and self.err >= self.optimizer_cfg.error_threshold:
            k, err, add = self._find_best_line(nail)
            self._update(k, add, err)
            progress.update(1 if add else -1)
            if callback_it is not None:
                self.canvas.add_fiber(*self.lines[-1])
                callback_it(it, self.canvas)
            it += 1
            if self.optimizer_cfg.continuous:
                line = self.nail_pairs[k]
                nail = line[0] if line[0] != nail else line[1]
            best_err = min(best_err, (it, err), key=lambda x: x[1])
        progress.close()

        # trim lines at best error
        # self.lines = self.lines[:best_err[0]]

        # get path
        if debug_path is not None:
            np.save(os.path.join(debug_path, DebugFilenames.LINES_VECTOR), self.x)
        if self.optimizer_cfg.continuous and self.lines:
            # arrange nails in the right order by choosing the other nail from each line
            lines = []
            cur_nail = self.lines[0][0] if self.lines[0][0] in self.lines[1] else self.lines[0][1]
            lines.append(cur_nail)
            for nail_pair in self.lines[1:]:
                cur_nail = nail_pair[0] if nail_pair[1] == cur_nail else nail_pair[1]
                lines.append(cur_nail)
        else:
            lines = copy(self.lines)
        return lines

    def _find_best_line(self, nail: Optional[int] = None) -> Tuple[int, float, bool]:
        """
        Find best line to add or remove, that minimizes the error.
        :param nail: if given, will check only valid lines to form a continuous path from nail.
        :return k: next line index to add or remove.
        :return err: updated error after adding/removing line.
        :return add: is the line to be added (True) or removed (False).
        """
        # get valid matrices and pre-calculate inner product
        if nail is not None:
            valid_lines = self.nail_to_lines[nail]
            if self.optimizer_cfg.n_random_nails is not None and self.optimizer_cfg.n_random_nails > 0:
                n = min(len(valid_lines), self.optimizer_cfg.n_random_nails)
                valid_lines = random.sample(valid_lines, n)
        else:
            valid_lines = list(range(len(self.x)))
            if self.optimizer_cfg.n_random_nails is not None and self.optimizer_cfg.n_random_nails > 0:
                # sample n_random nails per nail
                n_sample_lines = int(len(valid_lines) * self.optimizer_cfg.n_random_nails / self.canvas_cfg.nails)
                n = min(len(valid_lines), n_sample_lines)
                valid_lines = random.sample(valid_lines, n)
        A = self.A[:, valid_lines]
        norms = self.norms[valid_lines]
        x = self.x[valid_lines]
        inner_products = np.squeeze(np.asarray(self.residual @ A))  # (n_lines)

        # adding a line
        add_errs = self.err + norms + 2 * inner_products  # (n_lines)
        add_errs[x == 1] = np.inf  # can't add lines which were already added
        next_add = valid_lines[int(np.argmin(add_errs))]
        next_add_err = float(np.min(add_errs))

        # removing a line - only if not continuous
        if nail is None:
            remove_errs = self.err + norms - 2 * inner_products  # (n_lines)
            remove_errs[x == 0] = np.inf  # can't remove lines which weren't added
            next_remove = valid_lines[int(np.argmin(remove_errs))]
            next_remove_err = float(np.min(remove_errs))
        else:
            next_remove, next_remove_err = 0, float('inf')

        if next_add_err < next_remove_err:
            return next_add, next_add_err, True
        else:
            return next_remove, next_remove_err, False

    def _update(self, k: int, add: bool, err: float):
        """
        Add or remove a line, and update state.
        :param k: line index to update.
        :param add: add or remove this line.
        :param err: error achieved by this update.
        """
        if add:
            self.x[k] = 1
            self.residual = self.residual + np.squeeze(self.A[:, k].todense())
            self.n_fibers += 1
            self.lines.append(self.nail_pairs[k])
        else:
            self.x[k] = 0
            self.residual = self.residual - np.squeeze(self.A[:, k].todense())
            self.lines.remove(self.nail_pairs[k])
            self.n_fibers -= 1
        self.err = err

    def _setup(self, b: np.ndarray):
        """
        Initializes residuals for efficient error calculation:
            residual: (n_pixels) vector holding the current residual (r_k = Ax_k - b).
            err: current error (err_k = ||r_k||^2).
            norms: (n_lines) norms of each column of A.
            x: (n_lines) binary vector of which lines are used in solution.
            lines: lines used in current solution.
            n_fibers: number of lines used in current solution (sparsity of x).
        """
        self.residual = -b
        self.err = np.sum(self.residual ** 2)
        self.norms = np.array(self.A.power(2).sum(axis=0)).flatten()
        self.x = np.zeros(len(self.nail_pairs))
        self.lines = []
        self.n_fibers = 0
