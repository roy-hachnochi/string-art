import os
import sys

from string_art.utils import parse_args, init_config
from string_art.globals import DebugFilenames
from main import run_string_art, run_postprocess, run_preprocess


def main():
    params = {
        'bee': {'canvas_size': (600, 600), 'colors': ['#efac2a', '#e3e4de', '#000000', '#ffffff']},
        'blade_runner': {'canvas_size': (600, 600), 'colors': ['#ffffff', '#ff0000', '#1ee8fa', '#000000']},
        'cat': {'canvas_size': (400, 600), 'colors': ['#ffffff', '#40b975', '#a46445', '#000000']},
        'cat2': {'canvas_size': (440, 600), 'colors': ['#ffffff', '#cf5525', '#d0d000', '#000000']},
        'coraline': {'canvas_size': (400, 600), 'colors': ['#ffffff', '#ff32ff', '#3cc8ff', '#003ca0', '#7832c8', '#000000']},
        'duck': {'canvas_size': (470, 600), 'colors': ['#ffffff', '#ffff00', '#ff0000', '#000000']},
        'earth': {'canvas_size': (600, 600), 'colors': ['#ffffff', '#009600', '#0064ff', '#964100', '#000000']},
        'eye': {'canvas_size': (350, 600), 'colors': ['#ffffff', '#e29fa0', '#0070e0', '#000000']},
        'fish': {'canvas_size': (600, 600), 'colors': ['#ffffff', '#ff6400', '#3296dc', '#000000']},
        'fish2': {'canvas_size': (400, 600), 'colors': ['#ffffff', '#ff8200', '#000000']},
        'fox': {'canvas_size': (600, 450), 'colors': ['#ffffff', '#9e3a10', '#d5904f', '#000000']},
        'jellyfish': {'canvas_size': (600, 450), 'colors': ['#ffffff', '#f11672', '#007bff', '#000000']},
        'leopard': {'canvas_size': (400, 600), 'colors': ['#ffffff', '#ffa200', '#000000']},
        'lion': {'canvas_size': (600, 500), 'colors': ['#ffffff', '#ff9600', '#ff0000', '#000000']},
        'london_telephone_box': {'canvas_size': (600, 600), 'colors': ['#ffffff', '#ff0000', '#000000']},
        'mona_lisa': {'canvas_size': (600, 600), 'colors': ['#ffffff', '#e6c850', '#0032c8', '#006e32','#642800', '#000000']},
        'phoenix': {'canvas_size': (350, 600), 'colors': ['#ffa200', '#ff0000', '#0091ff', '#000000']},
        'pink_floyd': {'canvas_size': (400, 600), 'colors': ['#ffffff', '#ef2d44', '#fa9137', '#f8f720', '#46c765', '#8c7ca8', '#000000']},
        'planets': {'canvas_size': (330, 600), 'colors': ['#ffffff', '#e6c850', '#0064ff', '#ff0000', '#964100', '#000000']},
        'sauron': {'canvas_size': (300, 600), 'colors': ['#ffffff', '#ff0000', '#ff7000', '#ffff00', '#000000']},
        'snake': {'canvas_size': (400, 600), 'colors': ['#ffffff', '#ffff00', '#ff0000', '#000000']},
        'stag': {'canvas_size': (350, 600), 'colors': ['#ffffff`', '#00d7e1', '#0078f0', '#000078', '#000000']},
        'tiger': {'canvas_size': (600, 475), 'colors': ['#ffffff`', '#ff8200', '#ff0000', '#000000']},
        'union_jack': {'canvas_size': (600, 600), 'colors': ['#ffffff', '#ff0000', '#0078f0', '#000000']},
        'volcano': {'canvas_size': (400, 600), 'colors': ['#ffffff', '#ff2000', '#ffff00', '#000000']},
    }

    # params = [
    #     'don_draper',
    #     'einstein',
    #     'gatsby',
    #     'godfather',
    #     'joker',
    #     'kill_bill'
    #     'mona_lisa',
    #     'morrison',
    #     'pulp_fiction',
    #     'terminator',
    #     'walter_white',
    # ]

    name = 'MCBL_log'
    optimizer = 'multicolor_binary_linear'  # 'binary_linear'/'multicolor_binary_linear'/'greedy'/'LS'
    bg_color = '#ffffff'
    n_fibers = '10000'
    interval = '0.1'
    config = 'multicolor_config'  # 'multicolor_config'/'bw_config'

    for image in params:
        sys.argv = [
            'main.py',
            '--image', f'examples/{image}.jpg',
            # '--weights', 'images/planets_w.png',
            '--result', f'results/{image}',
            '--name', f'{image}_{name}',
            '--config', config,
            # '--shape', 'ellipse',
            '--fiber_width', '0.12',
            '--canvas_size'] + [str(s) for s in params[image]['canvas_size']] + [
            # '--optimization_resolution', '525', '900',
            '--bg_color', bg_color,
            '--n_fibers', n_fibers,
            # '--rgbcmykw',
            # '--n_colors', '4',
            '--colors'] + params[image]['colors'] + [
            '--nails', '360',
            '--optimizer_type', optimizer,
            # '--optimizer_type', 'binary_linear',
            # '--simulate_combine',
            '--interval', interval,
            # '--postprocess_only',
            # '--preprocess_only',
            # '--save_mp4',
            '--verbose',
            '--debug', f'results/{image}/{name}',
        ]

        # -----------------------------------------------------------------------
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
