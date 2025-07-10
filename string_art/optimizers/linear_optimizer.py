from typing import Optional
import os
from tqdm import tqdm
from copy import copy

import numpy as np
from scipy import sparse

from .base_optimizer import BaseOptimizer
from .fibers_path import vec_to_lines
from string_art.utils import IterationCallback
from string_art.globals import PATH_TYPE, DebugFilenames


class LinearModelOptimizer(BaseOptimizer):
    """
    Linear model optimizer - Model the string-art problem as a sparse linear optimization problem:
    min. ||Ax-b||. Where:
    - b is the flattened target image.
    - x is a binary vector stating which lines are to be added.
    - A is the transformation that sums all lines to an image.

    Solve it a linear solver:
    - Least-Squares - Regular least squares solver, enforce sparsity by thresholding.
    """

    MATRIX_PATH = 'matrix_cache'  # path to save the big constructed A so we don't have to construct it each time.

    def _optimize(
            self,
            image: np.ndarray,
            weights: Optional[np.ndarray] = None,
            n_fibers: Optional[int] = None,
            callback_it: Optional[IterationCallback] = None,
            debug_path: Optional[str] = None,
    ) -> PATH_TYPE:
        # optimize
        b = image.flatten()
        x = sparse.linalg.lsqr(self.A, b)[0]

        # convert to edges
        if debug_path is not None:
            np.save(os.path.join(debug_path, DebugFilenames.LINES_VECTOR), x)
        lines = vec_to_lines(x, self.nail_pairs, self.optimizer_cfg.threshold)
        return lines

    def _init(self, weights: Optional[np.ndarray] = None):
        """
        Load or construct the A matrix - the transformation that sums all lines to an image.
        Each column of A is a flattened image modeling a different line in the image.
        Constructs:
            A: (n_pixels, n_lines) transformation matrix from lines to (flattened) image.
            nail_pairs: (n_lines) array of tuples matching columns in A to lines represented by nail pairs.
            nail_to_lines: (n_nails) list with indices of valid lines for a continuous path for each nail.
        """
        # construct lines by nail pairs
        self.nail_pairs = copy(self.canvas.valid_lines)
        self.nail_to_lines = [[ind for ind, line in enumerate(self.canvas.valid_lines) if nail in line] for nail in
                              range(self.canvas_cfg.nails)]

        # if A saved - load it
        matrix_path = self._get_matrix_path()
        if os.path.exists(matrix_path):
            n_rows = self.canvas_cfg.resolution[0] * self.canvas_cfg.resolution[1]
            if not hasattr(self, 'A') or self.A.shape[0] != n_rows:
                # if A wasn't already initialized with the right config
                self.A = sparse.load_npz(matrix_path)

        # if not - construct it
        else:
            col_pointers = [0]
            row_indices = []
            data = []
            for line in tqdm(self.nail_pairs, desc='Building line-image transformation matrix'):
                (line_r, line_c), line_val = self.canvas.get_fiber(*line)
                flat_indices = np.ravel_multi_index((line_r, line_c), dims=self.canvas_cfg.resolution)
                row_indices.extend(flat_indices)
                data.extend(line_val)
                col_pointers.append(len(data))

            num_rows = self.canvas_cfg.resolution[0] * self.canvas_cfg.resolution[1]
            num_cols = len(self.nail_pairs)
            self.A = sparse.csc_matrix((data, row_indices, col_pointers), shape=(num_rows, num_cols))

            # save A for future use
            os.makedirs(self.MATRIX_PATH, exist_ok=True)
            sparse.save_npz(matrix_path, self.A)

        # add weights
        if weights is not None:
            self.A = self.A.multiply(weights.flatten()[:, np.newaxis]).tocsc()

    def _get_matrix_path(self):
        path = f'size_{self.canvas_cfg.size[0]}x{self.canvas_cfg.size[1]}'
        path += f'_res_{self.canvas_cfg.resolution[0]}x{self.canvas_cfg.resolution[1]}'
        path += f'_nails_{self.canvas_cfg.nails}'
        path += f'_shape_{self.canvas_cfg.shape.value}'
        path += f'_width_{self.canvas_cfg.fiber_width}'
        path += f'_type_{"constant" if self.canvas_cfg.fiber_constant else "AA"}'
        path += '.npz'
        return os.path.join(self.MATRIX_PATH, path)
