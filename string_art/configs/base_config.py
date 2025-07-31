from dataclasses import dataclass
from typing import Tuple, Optional, Union
from enum import Enum

from numpy import ndarray


class Shape(Enum):
    ELLIPSE = 'ellipse'
    RECTANGLE = 'rectangle'


class OptimizerType(Enum):
    GREEDY = 'greedy'
    LEAST_SQUARES = 'LS'
    BINARY_LINEAR = 'binary_linear'
    MULTICOLOR_BINARY_LINEAR = 'multicolor_binary_linear'
    RADON = 'radon'  # TODO: implement?
    NEURAL_NETWORK = 'NN'  # TODO: implement


class PaletteType(Enum):
    RGBCMYKW = 'rgbcmykw'
    PATCHES_SIMULATION = 'patches'
    HISTOGRAM = 'histogram'
    HISTOGRAM_AND_SIMULATION = 'histogram_and_patches'
    CLUSTERING = 'clustering'


@dataclass
class CanvasConfig:
    size: Tuple[int, int]  # real canvas size (h, w) [millimeters]
    nails: int  # number of nails around canvas
    shape: Union[Shape, str]  # shape of the canvas (Shape.ELLIPSE/Shape.RECTANGLE)
    fiber_width: float  # real fiber width [millimeters]
    resolution: Optional[Tuple[int, int]] = None  # canvas resolution (h, w) [pixels], if not given - multiply size such that fiber_width will turn out as 0.6 pixel
    fiber_constant: bool = False  # is fiber with constant values, if False - apply antialiasing
    bg_color: Tuple[int, int, int] = (255, 255, 255)  # background color of canvas (only for MulticolorCanvas)

    def __post_init__(self):
        self._updating = False  # this makes sure that we don't call _update methods while updating
        self.nails = 4 * (self.nails // 4)  # make number of nails divisible by 4
        self._fixed_resolution = self.resolution is not None  # re-update resolution only if it isn't fixed by user
        self._update_shape()
        self._update_resolution()
        self._initialized = True  # this makes sure that we only call _update methods after all fields were initialized

    def _update_resolution(self):
        self._updating = True
        if not self._fixed_resolution:
            # set resolution so that fiber_value will turn out as 0.6
            self.resolution = (int(self.size[0] / self.fiber_width * 0.6), int(self.size[1] / self.fiber_width * 0.6))
        self.fiber_value = min(1., self.fiber_width * self.resolution[0] / self.size[0],
                               self.fiber_width * self.resolution[1] / self.size[1])
        self._updating = False

    def _update_shape(self):
        self._updating = True
        if isinstance(self.shape, str):
            try:
                self.shape = next(s for s in Shape if s.value == self.shape)
            except StopIteration:
                print(f'\'{self.shape}\' is not a valid value for shape, see class Shape.')
        self._updating = False

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if hasattr(self, '_initialized') and self._initialized and not self._updating:
            if key in ['size', 'resolution', 'fiber_width']:
                if key == 'resolution':
                    self._fixed_resolution = True
                self._update_resolution()
            if key == 'shape':
                self._update_shape()


@dataclass
class PreprocessConfig:
    resolution: Optional[Tuple[int, int]] = None  # resize image to this resolution

    # for multicolor images
    colors: Optional[ndarray] = None  # manual palette for dithering
    palette_type: Optional[PaletteType] = PaletteType.HISTOGRAM_AND_SIMULATION  # palette selection method (if not provided manual palette)
    n_colors: int = 4  # number of colors to use for dithering palette

    def __post_init__(self):
        self._update_palette_type()
        assert not (self.colors is None and self.palette_type is None), \
            ('No palette provided. Either set colors for manual palette, or set palette_type for automatic palette '
             'calculation method.')

    def _update_palette_type(self):
        if isinstance(self.palette_type, str):
            try:
                self.palette_type = next(s for s in PaletteType if s.value == self.palette_type)
            except StopIteration:
                if self.colors is None:
                    print(f'\'{self.palette_type}\' is not a valid value for palette_type, see class PaletteType.')

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'palette_type':
            self._update_palette_type()


@dataclass
class OptimizerConfig:
    # Shared
    n_fibers: int  # max number of fibers to run through the canvas
    type: Union[OptimizerType, str]  # optimizer type to run
    multicolor: bool  # multicolor image optimization
    error_threshold: float = 0  # sufficient optimization error, stops if reached it during optimization

    # Greedy/BinaryLinear optimizer
    continuous: bool = True  # enforce forming a continuous path between nails during optimization
    n_random_nails: Optional[int] = None  # restrict connection to a random subset of nails each iteration, for efficiency

    # Least squares optimizer
    threshold: float = 0.25  # threshold to set fiber as 1 above and 0 below

    # Postprocessing
    simulate_combine: bool = False  # instead of interweaving with fixed interval, combine colors by importance (by error simulation)
    interval: float = 0.25  # interweaving interval to switch between colors when combining (0 < interval <= 1), only for simulate_combine = False

    def __post_init__(self):
        self._update_type()

    def _update_type(self):
        if isinstance(self.type, str):
            try:
                self.type = next(s for s in OptimizerType if s.value == self.type)
            except StopIteration:
                print(f'\'{self.type}\' is not a valid value for shape, see class OptimizerType.')

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'type':
            self._update_type()


@dataclass
class Config:
    canvas: CanvasConfig
    preprocess: PreprocessConfig
    optimizer: OptimizerConfig
