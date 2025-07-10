from string_art.configs.base_config import Config, CanvasConfig, PreprocessConfig, OptimizerConfig, Shape, OptimizerType


BW_CONFIG = Config(
    canvas=CanvasConfig(
        size=(600, 600),
        nails=240,
        shape=Shape.ELLIPSE,
        fiber_width=0.12,
        resolution=None,
        fiber_constant=False,
    ),
    preprocess=PreprocessConfig(
        resolution=None,
        ),
    optimizer=OptimizerConfig(
        n_fibers=3000,
        type=OptimizerType.BINARY_LINEAR,
        multicolor=False,
        error_threshold=0,
        continuous=True,
        n_random_nails=150,
        threshold=0.25,
    )
)
