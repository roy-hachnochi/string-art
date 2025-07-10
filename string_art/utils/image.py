from typing import Tuple

from PIL import Image
import numpy as np


def load_image(path: str, grayscale: bool = True) -> np.ndarray:
    """
    Load an image.
    :param path: image path.
    :param grayscale: grayscale or RGB.
    :return: image, either (3, h, w) for RGB or (h, w) for Grayscale, scaled to [0, 1].
    """
    image = Image.open(path)
    if grayscale:
        image = image.convert('L')
    img = np.array(image, dtype=np.float32) / 255.
    if not grayscale:
        img = np.transpose(img, (2, 0, 1))[:3]
    return img


def save_image(img: np.ndarray, path: str):
    """
    Save an image (scaled to [0, 1]).
    :param img: image to save.
    :param path: image path.
    """
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    image = Image.fromarray((img * 255).astype(np.uint8))
    image.save(path)


def resolution_by_aspect_ratio(resolution: Tuple[int ,int], aspect: float = 1.) -> Tuple[int ,int]:
    """
    Calculate new resolution of image based on desired aspect ratio (by contracting longer axis).
    :param resolution: (h, w) original resolution.
    :param aspect: desired aspect ratio (height / width).
    :return: (h, w) new resolution.
    """
    orig_aspect = resolution[0] / resolution[1]
    if aspect < orig_aspect:  # contract height axis
        return int(aspect * resolution[1]), resolution[1]
    elif aspect > orig_aspect:  # contract width axis
        return resolution[0], int(resolution[0] / aspect)
    return resolution


def rgb2hex(rgb: Tuple[int, int, int]) -> str:
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def hex2rgb(hex: str) -> Tuple[int, int, int]:
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
