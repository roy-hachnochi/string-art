from argparse import ArgumentParser
from dataclasses import is_dataclass

import numpy as np

from string_art.configs import CanvasConfig, OptimizerConfig, PreprocessConfig, Config, Shape, OptimizerType, get_config
from string_art.utils import hex2rgb

def parse_args():
    parser = ArgumentParser(description="String Art Optimization")

    # Input arguments
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--result', type=str, required=True, help='Result folder path')
    parser.add_argument('--weights', type=str, help='Path to optimization weights image, black = high white = low (optional)')
    parser.add_argument('--config', type=str, help='Name of predefined config in string_art/configs (optional)')
    parser.add_argument('--debug', type=str, help='Debug folder path (optional)')
    parser.add_argument('--name', type=str, help='Name of experiment/image (optional)')
    parser.add_argument('--verbose', action='store_true', help='Verbose prints during optimization')
    parser.add_argument('--preprocess_only', action='store_true', help='Only perform preprocessing (to observe target image before optimization)')
    parser.add_argument('--postprocess_only', action='store_true', help='Only perform postprocessing (use only for multicolor - to choose color paths combination method after optimization)')
    parser.add_argument('--save_mp4', action='store_true', help='Save MP4 of string-art rendering process')
    parser.add_argument('--plot_result', action='store_true', help='Show result when finished')

    # CanvasConfig arguments
    parser.add_argument('--canvas_size', type=int, nargs=2, help='Canvas size (h, w) in mm')
    parser.add_argument('--nails', type=int, help='Number of nails around canvas')
    parser.add_argument('--shape', type=str, choices=[s.value for s in Shape], help=f'Shape of the canvas ({"/".join([s.value for s in Shape])})')
    parser.add_argument('--fiber_width', type=float, help='Real fiber width in mm')
    parser.add_argument('--fiber_constant', action='store_const', const=True, default=None, help='Use constant fiber simulation instead of antialiasing fiber')
    parser.add_argument('--bg_color', type=str, default='#ffffff', help='Background color (HEX)')

    # PreprocessConfig arguments
    parser.add_argument('--optimization_resolution', type=int, nargs=2, help='Optimization canvas resolution (h, w)')
    parser.add_argument('--colors', type=str, nargs='+', help="Manual palette, list of HEX colors of colors to use")
    parser.add_argument('--rgbcmykw', action='store_const', const=True, default=None, help='Use RGBCMYKW subset as palette')
    parser.add_argument('--n_colors', type=int, default=4, help='Number of colors to use for dithering palette')

    # OptimizerConfig arguments
    parser.add_argument('--n_fibers', type=int, help='Max number of fibers in the canvas')
    parser.add_argument('--optimizer_type', type=str, choices=[s.value for s in OptimizerType], help=f'Type of optimizer to use ({"/".join([o.value for o in OptimizerType])})')
    parser.add_argument('--multicolor', action='store_const', const=True, default=None, help='Apply multicolor optimization')
    parser.add_argument('--error_threshold', type=float, default=0, help='Sufficient error threshold to halt during optimization')
    parser.add_argument('--noncontinuous', dest='continuous', action='store_const', const=False, default=True, help='Don\'t enforce continuous path optimization')
    parser.add_argument('--n_random_nails', type=int, help='Limit connections to random subset of nails each iteration')
    parser.add_argument('--threshold', type=float, default=0.25, help='Threshold for setting fiber values in least squares optimizer')
    parser.add_argument('--simulate_combine', action='store_true', help='Instead of interweaving with fixed interval, combine colors by importance (by error simulation)')
    parser.add_argument('--interval', type=float, default=0.3, help='Interweaving interval to switch between colors when combining (0 < interval <= 1), only for simulate_combine = False')

    return parser.parse_args()


def init_config(args):
   # init config
    if args.config:
        config = get_config(args.config)
    else:
        assert args.canvas_size is not None and \
               args.nails is not None and \
               args.shape is not None and \
               args.fiber_width is not None and \
               args.n_fibers is not None and \
               args.optimizer_type is not None, 'No config name received, make sure to set all necessary config args (canvas_size, nails, shape, fiber_width, n_fibers, optimizer_type).'
        config = Config(
            canvas=CanvasConfig(size=tuple(args.canvas_size), nails=args.nails, shape=args.shape,
                                fiber_width=args.fiber_width),
            preprocess=PreprocessConfig(),
            optimizer=OptimizerConfig(n_fibers=args.n_fibers, type=args.optimizer_type, multicolor=args.multicolor),
        )

    # update forced args
    for key, val in vars(args).items():
        keys = _key_mapping(key)
        if keys is None or val is None:  # not a config arg, or didn't receive in argparser
            continue
        if not _set_config_param(config, keys, _convert_value(key, val)):
            raise ValueError(f'Member {keys} not found in config.')
    return config


def _key_mapping(key):
    mappings = {
        'image': None,
        'result': None,
        'weights': None,
        'config': None,
        'debug': None,
        'name': None,
        'verbose': None,
        'preprocess_only': None,
        'postprocess_only': None,
        'save_mp4': None,
        'plot_result': None,
        'canvas_size': 'canvas.size',
        'optimization_resolution': 'preprocess.resolution',
        'optimizer_type': 'optimizer.type',
    }
    return mappings.get(key, key)


def _convert_value(key, val):
    """
    Converts a config value received from argparser to its correct value type.
    :param key: argparser key of the value to convert.
    :param val: value to convert.
    :return: new value
    """
    if key == 'colors':  # hex -> np array
        return np.array([hex2rgb(color) for color in val], dtype=np.float32) / 255.
    if key == 'bg_color':  # hex -> rgb
        return hex2rgb(val)
    if isinstance(val, list):
        return tuple(val)
    return val


def _set_config_param(config, key, val):
    """
    Traverse config to find keys path, and update the members value.
    :param config: base config to update.
    :param key: string defining a key path of the member to update. For example:
        - 'subconfig1.key1' will search for [any nested subconfigs].subconfig1.key1 somewhere in the config.
        - 'key1' will search for [any nested subconfigs].key1 somewhere in the config.
    :param val: value to update.
    :return: if the member was found and updated.
    """
    keys = key.split('.')
    if hasattr(config, keys[0]):  # found key -> update
        if len(keys) == 1:  # last key -> update
            setattr(config, keys[0], val)
            return True
        else:  # not last key -> continue searching for next keys recursively
            return _set_config_param(getattr(config, keys[0]), '.'.join(keys[1:]), val)
    elif is_dataclass(config):  # search recursively for key in subconfigs
        for subconfig in vars(config).values():
            if _set_config_param(subconfig, key, val):
                return True
        return False
    else:  # found a member and not a subconfig
        return False
