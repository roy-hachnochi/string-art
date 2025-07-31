from string_art.configs.base_config import (Config, CanvasConfig, PreprocessConfig, OptimizerConfig, Shape,
                                            OptimizerType, PaletteType)


MULTICOLOR_CONFIG = Config(
    canvas=CanvasConfig(
        size=(600, 600),
        nails=360,
        shape=Shape.RECTANGLE,
        fiber_width=0.12,
        resolution=None,
        fiber_constant=False,
        bg_color=(255, 255, 255),
    ),
    preprocess=PreprocessConfig(
        resolution=None,
        colors=None,
        palette_type=PaletteType.HISTOGRAM_AND_SIMULATION,
        n_colors=4,
        ),
    optimizer=OptimizerConfig(
        n_fibers=10000,
        type=OptimizerType.BINARY_LINEAR,
        multicolor=True,
        error_threshold=0,
        continuous=True,
        n_random_nails=150,
        threshold=0.25,
    )
)
