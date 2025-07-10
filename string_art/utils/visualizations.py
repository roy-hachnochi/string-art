from typing import Dict, Tuple, Optional
import sys
import os
import numpy as np

from .image import save_image
from string_art.globals import COLOR_IMAGES_TYPE, COLOR_TYPE, LINES_TYPE
from string_art.canvas import MulticolorCanvas
from string_art.configs.base_config import CanvasConfig


_WHITE_THRESHOLD = 235
COLORBAR_PLACEHOLDER = '**cbar**'

def _enable_ansi_windows():
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
            kernel32.SetConsoleMode(handle, 7)
        except Exception:
            pass  # Fail silently if enabling fails

_enable_ansi_windows()

def visualize_color_images(color_images: COLOR_IMAGES_TYPE, path: Optional[str] = None, force_horizontal: bool = False):
    """
    Concatenate multiple single-color images and save them together.
    :param color_images: dict of (h, w) image per color (scaled to [0, 1]), color is a 3-tuple of RGB values in [0, 255].
    :param path: optional path to save output image.
    :param force_horizontal: force horizontal concatenation of images, otherwise concatenates on shorter axis.
    """
    color_images_list = []
    for color, color_image in color_images.items():
        color_np = np.array(color)[:, np.newaxis, np.newaxis] / 255.
        if np.mean(color_np) <= _WHITE_THRESHOLD / 255.:  # use white BG
            white = np.array([1., 1., 1.])[:, np.newaxis, np.newaxis]
            color_image = color_image[np.newaxis, :, :] * color_np + (1 - color_image[np.newaxis, :, :]) * white
        else:  # color is near-white -> use black BG
            color_image = color_image[np.newaxis, :, :] * color_np
        color_images_list.append(color_image)
    # concatenate on longer axis
    c, h, w = color_images_list[0].shape
    axis = 2 if force_horizontal or h >= w else 1
    color_images_combined = np.concatenate(color_images_list, axis=axis)
    if path is not None:
        save_image(color_images_combined, path)
    return color_images_combined


def render_color_paths(
        color_paths: Dict[COLOR_TYPE, LINES_TYPE],
        canvas_cfg: CanvasConfig,
        save_path: Optional[str] = None,
        force_horizontal: bool = False,
):
    """
    Render multiple single-colored string-art images, concatenate and save them together.
    :param color_paths: dict of rendering path per color (path is a list of nails or a list of nail pairs, and color is
        a 3-tuple of RGB values in [0, 255]).
    :param canvas_cfg: configuration of canvas for rendering.
    :param save_path: optional path to save output image.
    :param force_horizontal: force horizontal concatenation of images, otherwise concatenates on shorter axis.
    """
    color_images_list = []
    for color, lines in color_paths.items():
        canvas = MulticolorCanvas(canvas_cfg)
        if color[0] + color[1] + color[2] >= _WHITE_THRESHOLD * 3:  # for near-white string use black BG
            bg_color = (0, 0, 0)
        else:
            bg_color = (255, 255, 255)
        color_path = [(color, lines)]
        canvas.set_bg_color(bg_color)
        canvas.render(color_path, opaque=False)
        color_images_list.append(canvas.get_image())
    # concatenate on longer axis
    c, h, w = color_images_list[0].shape
    axis = 2 if force_horizontal or h >= w else 1
    color_images_combined = np.concatenate(color_images_list, axis=axis)
    if save_path is not None:
        save_image(color_images_combined, save_path)
    return color_images_combined


def print_with_colorbar(template: str, rgb: Tuple[int, int, int], size: int = 10):
    r, g, b = rgb
    if "PYCHARM_HOSTED" in os.environ or sys.stdout.isatty():
        text = template.replace(COLORBAR_PLACEHOLDER, f'\033[48;2;{r};{g};{b}m{" " * size}\033[0m')
        print(text)
    elif "ipykernel" in sys.modules:
        from IPython.display import display, HTML
        width = int(size * 7)
        text = template.replace(COLORBAR_PLACEHOLDER, f'<span style="display:inline-block; width:{width}px; height:15px; background-color:rgb({r},{g},{b}); margin: 0 4px;"></span>')
        display(HTML(text))
    else:  # not supported - print without colorbar
        text = template.replace(COLORBAR_PLACEHOLDER, '')
        print(text)
