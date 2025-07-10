import os
from typing import Union, Optional, Tuple
from itertools import combinations
import warnings

import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image

from string_art.utils import load_image, resolution_by_aspect_ratio
from string_art.utils.visualizations import visualize_color_images
from string_art.globals import COLOR_TYPE, COLOR_IMAGES_TYPE


RGBCMYKW = np.array([[0., 0., 0.],  # K - black
                     [1., 1., 1.],  # W - white
                     [1., 0., 0.],  # R - red
                     [0., 1., 0.],  # G - green
                     [0., 0., 1.],  # B - blue
                     [1., 1., 0.],  # Y - yellow
                     [1., 0., 1.],  # M - magenta
                     [0., 1., 1.]])  # C - cyan


def preprocess_image(
        image: Union[str, np.ndarray],
        resolution: Optional[Tuple[int, int]] = None,
        aspect_ratio: Optional[float] = None,
        grayscale: bool = False,
        colors: Optional[np.ndarray] = None,
        rgbcmykw: bool = False,
        n_colors: int = 4,
        bg_color: COLOR_TYPE = (128, 128, 128),
) -> Union[np.ndarray, Tuple[COLOR_IMAGES_TYPE, np.ndarray]]:
    """
    Load and preprocess an image.
    For grayscale images, will make black the high-intensity color instead of white.
    For colored images, will perform dithering and return a grayscale intensity image per color.
    :param image: either a path to an image or a (h, w) or (c, h, w) array representing an image (scaled to [0, 1]).
    :param resolution: if given, resize image to this resolution.
    :param aspect_ratio: if given, resize image by this aspect ratio (is overridden by resolution if given).
    :param grayscale: grayscale or RGB.
    :param colors: optional (n, c) array representing RGB colors to be used for dithering.
    :param rgbcmykw: use RGBCMYKW colors for dithering (use only one between colors or rgbcmyk).
    :param n_colors: if not given colors, will find an n_colors palette of image and use it for dithering.
                     if given rgbcmykw, will choose n_colors from RGBCMYKW, otherwise will calculate palette.
                     (must satisfy 2 <= n_colors <= 8).
    :return: (h, w) image for grayscale, or
             dict of (h, w) image per color (scaled to [0, 1]) + image dithered to these colors.
    """
    if grayscale:
        assert not rgbcmykw and colors is None, 'grayscale should be False when working with colored images.'
    else:
        assert not rgbcmykw or colors is None, 'Can\'t specify both colors and RGBCMYKW palette, use only one.'
        if colors is None:
            assert 1 <= n_colors <= 8, f'Must use at least 1 and at most 8 colors, received {n_colors}'

    if isinstance(image, str):
        image = load_image(image, grayscale=grayscale)

    # resize and fix resolution
    if aspect_ratio is not None and resolution is None:
        resolution = resolution_by_aspect_ratio(image.shape[-2:], aspect_ratio)
    if resolution is not None and image.shape[-2:] != resolution:
        if grayscale:
            image = cv2.resize(image, (resolution[1], resolution[0]))
        else:
            image = cv2.resize(image.transpose((1, 2, 0)), (resolution[1], resolution[0])).transpose((2, 0, 1))

    # handle grayscale - if loaded - flip black & white, otherwise just return image
    if grayscale:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return 1 - image

    # if colored - find palette, apply dithering and get image per color
    assert image.ndim == 3, 'Received colored image params but image is grayscale.'
    if rgbcmykw:
        colors = RGBCMYKW.copy()
        colors = np.concatenate((np.array([bg_color]) / 255., colors), axis=0)  # add BG color
        colors = _choose_best_palette_colors(image, colors, n_colors + 1, 3)  # fix BG + white + black
    elif colors is None:
        warnings.warn('Automatically computing palette may result in degraded results. Recommending to use '
                      'fixed palette (rgbcmykw=True) or manual palette (colors).', category=UserWarning)
        # TODO: doesn't work good, find a way to get the best palette (will probably involve some segmentation by color)
        colors = _get_palette_kmeans(image, n_colors, force_bw=False)
    if not rgbcmykw:
        colors = np.concatenate((np.array([bg_color]) / 255., colors), axis=0)  # add BG color

    # TODO: divide rows to batches for quicker dithering?
    image = np.expand_dims(image, axis=0)
    dithered = floyd_steinberg_dithering(image, colors)
    dithered = np.squeeze(dithered)
    color_images = {}
    for color_i in range(1, colors.shape[0]):  # skip BG color
        color_image = 1. * np.all(dithered == np.expand_dims(colors[color_i], (1, 2)), axis=0)  # (h, w)
        color_image = cv2.GaussianBlur(color_image, (5, 5), 0)
        color_tuple = tuple([int(255 * x) for x in colors[color_i]])
        color_images[color_tuple] = color_image
    return color_images, dithered


def floyd_steinberg_dithering(image: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """
    Perform Floyd-Steinberg dithering on image.
    :param image: (b, c, h, w) batch of images to perform dithering on (scaled to [0, 1]).
    :param colors: (n, c) colors for clustering (scaled to [0, 1]).
    :return dithered: (b, c, h, w) batch of dithered images where each pixel is only one of n colors.
    """
    fs_kernel = np.array([[0., -16., 7.], [3., 5., 1.]]) / 16
    b, c, h, w = image.shape
    padded_image = np.zeros((b, c, h + 1, w + 2))
    padded_image[:, :, :-1, 1:-1] = image.copy()

    # register pixels to nearest colors and propagate error
    for ri in range(h):
        for ci in range(1, w + 1):
            new_color = _find_nearest_color(padded_image[:, :, ri, ci], colors)  # (b, c)
            err = padded_image[:, :, ri, ci] - new_color  # (b, c)
            padded_image[:, :, ri:(ri + 2), (ci - 1):(ci + 2)] += np.expand_dims(err, (2, 3)) * fs_kernel
    dithered = padded_image[:, :, :-1, 1:-1]  # (b, c, h, w)

    # finally, re-register all pixels to their nearest to avoid numerical errors
    pixels = np.reshape(dithered.transpose((0, 2, 3, 1)), (-1, c))  # (bhw, c)
    pixels = _find_nearest_color(pixels, colors)  # (bhw, c)
    dithered = np.reshape(pixels, (b, h, w, c)).transpose((0, 3, 1, 2))  # (b, c, h, w)
    return dithered


# def ordered_dithering(image: np.ndarray, colors: np.ndarray, matrix_size: int = 4) -> np.ndarray:
#     """
#     Perform ordered dithering on image.
#     :param image: (b, c, h, w) batch of images to perform dithering on (scaled to [0, 1]).
#     :param colors: (n, c) colors for clustering (scaled to [0, 1]).
#     :param matrix_size: size of Bayer matrix.
#     :return dithered: (b, c, h, w) batch of dithered images where each pixel is only one of n colors.
#     """
#     b, c, h, w = image.shape
#     image = image.transpose(0, 2, 3, 1)  # shape (b, h, w, 3)
#
#     # Generate the Bayer matrix and tile it over the image.
#     bayer = _bayer_matrix(matrix_size)  # (matrix_size, matrix_size)
#     rep_y = int(np.ceil(h / matrix_size))
#     rep_x = int(np.ceil(w / matrix_size))
#     threshold_map = np.tile(bayer, (rep_y, rep_x))[:h, :w]  # (h, w)
#
#     # alpha controls the strength of dithering; alpha=1 gives perturbations in [-0.5, 0.5].
#     alpha = 0.5
#     adjustment = alpha * (threshold_map[None, :, :, None] - 0.5)  # (1, h, w, 1)
#     dithered_image = np.clip(image + adjustment, 0, 1)
#
#     # Flatten all pixels across the batch for quantization.
#     flat_pixels = dithered_image.reshape(-1, 3)  # shape (b * h * w, 3)
#     quantized_pixels = _find_nearest_color(flat_pixels, colors)  # (b*h*w, 3)
#     quantized_image = quantized_pixels.reshape(b, h, w, 3).transpose(0, 3, 1, 2)
#
#     return quantized_image
#
#
# def _bayer_matrix(n):
#     """
#     Return an n×n Bayer threshold matrix normalized to [0,1].
#     Supports n = 2, 4, or 8.
#     """
#     if n == 2:
#         return np.array([[0, 2],
#                          [3, 1]], dtype=float) / 4.0
#     elif n == 4:
#         return np.array([
#             [0,  8,  2, 10],
#             [12, 4, 14, 6],
#             [3, 11, 1, 9],
#             [15, 7, 13, 5]], dtype=float) / 16.0
#     elif n == 8:
#         base = _bayer_matrix(4)
#         return np.block([
#             [4*base + 0, 4*base + 2],
#             [4*base + 3, 4*base + 1]
#         ]) / 64.0
#     else:
#         raise ValueError("Unsupported Bayer matrix size. Use 2, 4, or 8.")


def _find_nearest_color(pixels: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """
    Find nearest color in array of colors.
    :param pixels: (b, c) batch of pixel values.
    :param colors: (n, c) colors for clustering.
    :return new_pixels: (b, c) batch of new colors where each pixel was changed to its nearest color.
    """
    # pixels_lab = rgb2lab(pixels.reshape(-1, 1, 1, 3)).reshape(-1, 3)
    # colors_lab = rgb2lab(colors.reshape(-1, 1, 1, 3)).reshape(-1, 3)
    # diffs = np.sum((np.expand_dims(pixels_lab, 1) - np.expand_dims(colors_lab, 0)) ** 2, axis=2)  # (b, n)
    diffs = np.sum((np.expand_dims(pixels, 1) - np.expand_dims(colors, 0)) ** 2, axis=2)  # (b, n)
    new_inds = np.argmin(diffs, axis=1)  # (b)
    new_pixels = colors[new_inds]  # (b, c)
    return new_pixels


def _get_palette_kmeans(image: np.ndarray, k: int, force_bw: bool = True) -> np.ndarray:
    """
    Find palette of most dominant colors an image (by K-Means clustering).
    :param image: (c, h, w) image (scaled to [0, 1]).
    :param k: number of dominant colors for clustering.
    :param force_bw: force black and white to be part of the palette as two of the cluster centers.
    :return palette: (k, c) palette of k dominant colors in image.
    """
    c, h, w = image.shape
    pixels = image.transpose((1, 2, 0)).reshape((-1, c))

    fixed_colors = np.array([[0., 0., 0.],
                             [1., 1., 1.]])
    n_fixed = fixed_colors.shape[0]
    if force_bw:
        random_centroids = pixels[np.random.choice(pixels.shape[0], k - n_fixed, replace=False)]
        initial_centroids = np.concatenate([fixed_colors, random_centroids], axis=0)
    else:
        initial_centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]

    kmeans = KMeans(n_clusters=k, init=initial_centroids, n_init=1)
    kmeans.fit(pixels)

    # ensure fixed colors remain as centroids
    palette = kmeans.cluster_centers_
    if force_bw:
        palette[:n_fixed, :] = fixed_colors

    # sort by descending cluster size
    cluster_sizes = np.bincount(kmeans.labels_)
    sorted_indices = np.argsort(cluster_sizes[n_fixed:])[::-1]
    palette[n_fixed:] = palette[n_fixed + sorted_indices]
    return palette


def _get_palette_median_cut(image: np.ndarray, k: int, force_bw: bool = True):
    """
    Find palette of most dominant colors an image (by Median-Cut algorithm).
    :param image: (c, h, w) image (scaled to [0, 1]).
    :param k: number of dominant colors for clustering.
    :param force_bw: (not used) force black and white to be part of the palette as two of the cluster centers.
    :return palette: (k, c) palette of k dominant colors in image.
    """
    image = Image.fromarray(np.transpose(image * 255, (1, 2, 0)).astype('uint8')).convert('RGB')
    image = image.quantize(colors=k, method=Image.MEDIANCUT)
    palette = np.array(image.getpalette()[:k * 3]).reshape(-1, 3) / 255.

    # sort by descending cluster size
    cluster_sizes = image.getcolors()  # a list of (count, color_index)
    sorted_indices = sorted(cluster_sizes, key=lambda x: x[0], reverse=True)
    sorted_indices = [ind for cluster_size, ind in sorted_indices]
    palette = palette[sorted_indices]
    return palette


def _choose_best_palette_colors(image: np.ndarray, colors: np.ndarray, n_colors: int, n_fixed: int) -> np.ndarray:
    """
    Given an image, a palette of n colors, a number k of fixed colors and a number m of desired palette colors, choose
    the best m-k out of the n colors to add to the palette to reconstruct the image. Best here is in the sense of
    minimal dithering error. This is done by simulating dithering on small random patches from the image.
    :param image: (c, h, w) image to reconstruct using the palette (scaled to [0, 1]).
    :param colors: (n, c) palette of n dominant colors in image.
    :param n_colors: m desired colors in the palette (m < n).
    :param n_fixed: k fixed colors in the palette (first k out of n, k < m).
    :return: (m, k) palette of m colors to reconstruct the image.
    """
    n_patches = 8
    patch_size = 128
    n_palette = colors.shape[0]
    c, h, w = image.shape
    assert n_fixed < n_colors <= colors.shape[0], (f'number of colors must satisfy n_fixed < n_colors <= n_palette, '
                                                  f'received ({n_fixed}, {n_colors}, {n_palette}).')
    assert patch_size <= min(h, w), f'patch_size ({patch_size}) must be smaller than image dims ({(h, w)}).'

    # create image patches
    h0_patches = np.random.randint(0, h - patch_size + 1, size=n_patches)
    w0_patches = np.random.randint(0, w - patch_size + 1, size=n_patches)
    patches = np.zeros((n_patches, c, patch_size, patch_size), dtype=image.dtype)
    blurred_patches = np.zeros((n_patches, c, patch_size, patch_size), dtype=image.dtype)
    for i in range(n_patches):
        patches[i] = image[:, h0_patches[i]:h0_patches[i] + patch_size, w0_patches[i]:w0_patches[i] + patch_size]
        blurred_patches[i] = cv2.GaussianBlur(patches[i].transpose((1, 2, 0)), (5, 5), 0).transpose((2, 0, 1))

    # try dithering fo all possible color combinations
    color_options = [list(comb) for comb in (combinations(list(range(n_fixed, n_palette)), n_colors - n_fixed))]
    try_palette = np.zeros((n_colors, c), dtype=colors.dtype)
    try_palette[:n_fixed] = colors[:n_fixed]
    min_err, best_comb = float('inf'), color_options[0]
    for color_comb in color_options:
        # dither, blur, and calc error
        try_palette[n_fixed:] = colors[color_comb]
        dithered_patches = floyd_steinberg_dithering(patches, try_palette)
        for patch_i in range(n_patches):
            dithered_patches[patch_i] = cv2.GaussianBlur(dithered_patches[patch_i].transpose((1, 2, 0)), (5, 5), 0).transpose((2, 0, 1))
        err = np.mean((dithered_patches - blurred_patches) ** 2)
        if err < min_err:
            min_err = err
            best_comb = color_comb

    # return new palette
    palette = np.zeros((n_colors, c), dtype=colors.dtype)
    palette[:n_fixed] = colors[:n_fixed]
    palette[n_fixed:] = colors[best_comb]
    return palette


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    from time import time
    from string_art.utils import save_image

    input_path = 'examples/fish.jpg'
    result_path = 'debug/fish_dithered.png'
    color_images_path = 'debug/fish_color_images.png'
    colors = np.array([[255., 255., 255.],
                       [255., 100., 0.],
                       [50., 150., 220.],
                       [0., 0., 0.]]) / 255.
    resolution = None
    rgbcmykw = False
    n_colors = 4

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    np.random.seed(42)
    tic = time()
    color_images, dithered = preprocess_image(input_path, colors=colors, rgbcmykw=rgbcmykw, n_colors=n_colors, resolution=resolution)
    preprocess_time = time() - tic
    print(f'Preprocess time: {preprocess_time:.2f}')
    save_image(dithered, result_path)
    visualize_color_images(color_images, color_images_path)
