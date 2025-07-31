from copy import deepcopy
from typing import Union, Optional, Dict
from time import time
import os
import random

import numpy as np

from string_art.configs import Config, get_config
from string_art.canvas import Canvas, MulticolorCanvas
from string_art.optimizers import optimizer_factory, FibersPath
from string_art.utils import (SaveCanvasCallback, SaveCanvasMulticolorCallback, SaveGifCallback, parse_args,
                              init_config, save_image, load_image)
from string_art.utils.visualizations import visualize_color_images, print_with_colorbar, COLORBAR_PLACEHOLDER
from string_art.globals import DebugFilenames, LINES_TYPE, COLOR_LINES_TYPE, COLOR_TYPE


def run_string_art(
        image: Union[str, np.ndarray],
        result_dir: str,
        weights: Optional[Union[str, np.ndarray]],
        config: Union[str, Config],
        debug_dir: Optional[str] = None,
        name: Optional[str] = None,
        verbose: bool = True,
        save_mp4: bool = False,
        plot_result: bool = False,
):
    # Initializations
    np.random.seed(42)
    random.seed(42)
    if isinstance(config, str):
        config = get_config(config)
    if debug_dir:
        if config.optimizer.multicolor:
            callback = SaveCanvasMulticolorCallback(300, debug_dir)
        else:
            callback = SaveCanvasCallback(300, debug_dir)
    else:
        callback = None

    # Optimize
    optimizer = optimizer_factory(config)
    lines = optimizer.optimize(image, weights=weights, callback_it=callback, debug_path=debug_dir, verbose=verbose)

    # Postprocess, render, and save result
    run_postprocess(image, lines, result_dir, config, debug_dir=debug_dir, name=name, save_mp4=save_mp4,
                    plot_result=plot_result)


def run_preprocess(
        image: Union[str, np.ndarray],
        result_dir: str,
        config: Union[str, Config],
        name: Optional[str] = None,
):
    # Initializations
    np.random.seed(42)
    random.seed(42)
    os.makedirs(result_dir, exist_ok=True)
    name = f'{name}_' if name is not None else ''
    if isinstance(config, str):
        config = get_config(config)

    # Preprocess
    tic = time()
    optimizer = optimizer_factory(config)
    preprocessed = optimizer.preprocess(image)
    preprocess_time = time() - tic
    print(f'Preprocess time: {preprocess_time:.2f}')
    if config.optimizer.multicolor:
        color_images, dithered = preprocessed
        print('Selected color palette:')
        for c in color_images:
            print_with_colorbar(f'{COLORBAR_PLACEHOLDER}', c)
        save_image(dithered, os.path.join(result_dir, name + DebugFilenames.DITHERED))
        visualize_color_images(color_images, os.path.join(result_dir, name + DebugFilenames.COLOR_IMAGES))
    else:
        save_image(preprocessed, os.path.join(result_dir, name + 'preprocessed.jpg'))


def run_postprocess(
        image: Union[str, np.ndarray],
        lines: Union[str, LINES_TYPE, COLOR_LINES_TYPE, Dict[COLOR_TYPE, LINES_TYPE]],
        result_dir: str,
        config: Union[str, Config],
        debug_dir: Optional[str] = None,
        name: Optional[str] = None,
        save_mp4: bool = False,
        plot_result: bool = False
):
    os.makedirs(result_dir, exist_ok=True)
    name = f'{name}_' if name is not None else ''

    # Postprocess
    lines_obj = FibersPath(lines)
    lines_obj.convert_lines_to_path(config.canvas.nails)
    if config.optimizer.simulate_combine:
        lines_obj.combine_color_paths(image, config.canvas)
    else:
        lines_obj.interweave_color_paths(config.optimizer.interval)
    lines_obj.trim_lines(config.optimizer.n_fibers)

    # Render
    if save_mp4:
        mp4_path = os.path.join(result_dir, name + DebugFilenames.RESULT_MP4)
        mp4_callback = SaveGifCallback(20, mp4_path, resolution=config.canvas.resolution)
    else:
        mp4_callback = None
    if debug_dir is not None:
        _save_debug_result(config, image, lines_obj, debug_dir, name)
    canvas = MulticolorCanvas(config.canvas) if config.optimizer.multicolor else Canvas(config.canvas)
    canvas.render(lines_obj.lines, callback=mp4_callback, max_fibers=config.optimizer.n_fibers)
    image_path = os.path.join(result_dir, name + DebugFilenames.RESULT)
    canvas.save(image_path)
    if save_mp4:
        mp4_callback.close()
    lines_obj.save_instructions(os.path.join(result_dir, name + DebugFilenames.INSTRUCTIONS), config.canvas, image_path)
    if plot_result:
        canvas.plot(show_nails=True)


def _save_debug_result(config: Config, image: Union[str, np.ndarray], lines_obj: FibersPath, debug_dir: str, name: str):
    config = deepcopy(config)
    if isinstance(image, str):
        image = load_image(image)
    config.canvas.resolution = config.preprocess.resolution if config.preprocess.resolution is not None else image.shape[-2:]
    if config.optimizer.multicolor:
        canvas = MulticolorCanvas(config.canvas, for_optimization=False)
        canvas.render(lines_obj.lines, max_fibers=config.optimizer.n_fibers, opaque=False)
    else:
        canvas = Canvas(config.canvas, for_optimization=False)
        canvas.render(lines_obj.lines, max_fibers=config.optimizer.n_fibers)
    image_path = os.path.join(debug_dir, name + DebugFilenames.RESULT)
    canvas.save(image_path)


def main():
    args = parse_args()
    config = init_config(args)
    if args.preprocess_only:
        run_preprocess(args.image, args.result, config, name=args.name)
    elif args.postprocess_only:
        if config.optimizer.multicolor:
            lines_path = os.path.join(args.debug, DebugFilenames.PER_COLOR_FIBERS)
        else:
            lines_path = os.path.join(args.debug, DebugFilenames.FIBERS)
        run_postprocess(args.image,
                        lines_path,
                        args.result,
                        config,
                        debug_dir=args.debug,
                        name=args.name,
                        save_mp4=args.save_mp4,
                        plot_result=args.plot_result)
    else:
        run_string_art(args.image,
                       args.result,
                       weights=args.weights,
                       config=config,
                       debug_dir=args.debug,
                       name=args.name,
                       verbose=args.verbose,
                       save_mp4=args.save_mp4,
                       plot_result=args.plot_result)


if __name__ == '__main__':
    main()
