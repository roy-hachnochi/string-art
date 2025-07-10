from typing import Union, List, Tuple, Dict
from dataclasses import dataclass
from numpy import ndarray


CONTINUOUS_PATH_TYPE = List[int]
PATH_TYPE = List[Tuple[int, int]]
LINES_TYPE = Union[CONTINUOUS_PATH_TYPE, PATH_TYPE]

COLOR_TYPE = Tuple[int, int, int]
COLOR_IMAGES_TYPE = Dict[COLOR_TYPE, ndarray]
COLOR_LINES_TYPE = List[Tuple[COLOR_TYPE, LINES_TYPE]]

MAX_FIBER_VAL = 0.2  # maximal fiber value for optimization, larger than this may be harder to optimize
SEQUENCE_LENGTH = 10  # add lines of same color in blocks of this size
MIN_SEQUENCE = 50  # if decided on a color, must add at least this number of lines from this color before switching

@dataclass
class DebugFilenames:
    LINES_VECTOR = 'lines_vec.npy'  # vector of values in [0, 1] of size (# of lines) indicating value of each line in the solution, before thresholding
    DITHERED = 'dithered.png'  # dithered image after preprocessing
    COLOR_IMAGES = 'color_images.jpg'  # single-color images after preprocessing (separated from dithered image)
    COLOR_RESULTS = 'color_results.jpg'  # intermediate per-color string-art results
    PER_COLOR_FIBERS = 'per_color_fibers.pkl'  # string-art path per color, before combining to a single multicolor path
    FIBERS = 'fibers.pkl'  # string-art path
    RESULT = 'string_art.jpg'  # final string-art rendering result
    RESULT_MP4 = 'string_art.mp4'  # MP4 of string-art rendering process
    INSTRUCTIONS = 'string_art_instructions.pdf'  # instructions of how to manually create the string-art
