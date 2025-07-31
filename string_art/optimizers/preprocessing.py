import os
from typing import Union, Optional, Tuple
from itertools import combinations

import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.color import hsv2rgb, rgb2lab
from PIL import Image

from string_art.utils import load_image, resolution_by_aspect_ratio
from string_art.utils.visualizations import visualize_color_images
from string_art.globals import COLOR_TYPE, COLOR_IMAGES_TYPE
from string_art.configs import PaletteType


RGBCMYKW = np.array([[0., 0., 0.],  # K - black
                     [1., 1., 1.],  # W - white
                     [1., 0., 0.],  # R - red
                     [0., 1., 0.],  # G - green
                     [0., 0., 1.],  # B - blue
                     [1., 1., 0.],  # Y - yellow
                     [1., 0., 1.],  # M - magenta
                     [0., 1., 1.]])  # C - cyan


COLOR_DICT = np.array([
    [0., 0., 0.],  # black
    [1., 1., 1.],  # white
    [0.5, 0.5, 0.5],  # gray
    [0.5, 0., 0.],  # dark red
    [1., 0.5, 0.5],  # light red
    [1., 0., 0.],  # red
    [1., 0.5, 0.],  # orange
    [0.6, 0.36, 0.],  # brown
    [1., 1., 0.],  # yellow
    [1., 1., 0.5],  # skin / pale yellow
    [0., 0.5, 0.],  # dark green
    [0., 1., 0.],  # green
    [0.25, 0.9, 0.8],  # turquoise
    [0., 1., 1.],  # cyan
    [0., 0.5, 0.5],  # dark cyan
    [0.2, 0.6, 0.85],  # pale blue
    [0., 0.47, 0.94],  # mid blue
    [0., 0., 1.],  # blue
    [0., 0., 0.47],  # dark blue
    [0.6, 0., 1.0],  # pale purple
    [0.6, 0., 0.47],  # dark purple
    [1., 0.5, 1.],  # pale pink
    [1., 0., 1.],  # magenta
    [0.95, 0.1, 0.45],  # pink
])


def preprocess_image(
        image: Union[str, np.ndarray],
        resolution: Optional[Tuple[int, int]] = None,
        aspect_ratio: Optional[float] = None,
        grayscale: bool = False,
        colors: Optional[np.ndarray] = None,
        palette_type: PaletteType = PaletteType.HISTOGRAM_AND_SIMULATION,
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
    :param palette_type: automatic palette selection method, if colors was not provided.
                         - RGBCMYKW - choose the best palette from RGBCMYKW color dictionary.
                         - PATCHES_SIMULATION - choose the best palette from COLOR_DICT, by calculating dithering error
                                                on small image patches.
                         - HISTOGRAM - choose palette based on image histogram.
                         - HISTOGRAM_AND_SIMULATION - use histogram method to reduce COLOR_DICT for better efficiency,
                                                      then use patches simulation method to choose the best palette.
                         - CLUSTERING - calculate palette via color clustering methods.
    :param n_colors: number of colors in automatic palette (must satisfy 1 <= n_colors <= 8).
    :param bg_color: background color of canvas.
    :return: (h, w) image for grayscale, or
             dict of (h, w) image per color (scaled to [0, 1]) + image dithered to these colors.
    """
    if grayscale:
        assert colors is None, 'grayscale should be False when working with colored images.'
    elif colors is None:
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
    if colors is None:
        colors = calculate_palette(image, palette_type, n_colors, bg_color)
    else:
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


def calculate_palette(
        image: np.ndarray,
        palette_type: PaletteType,
        n_colors: int,
        bg_color: COLOR_TYPE = (128, 128, 128),
) -> np.ndarray:
    """
    Calculate representing palette for image. This is a factory function for palette calculation methods.
    :param image: (c, h, w) image to reconstruct using the palette (scaled to [0, 1]).
    :param palette_type: palette selection method (see PaletteType for options).
    :param n_colors: m - number of desired colors in the palette.
    :param bg_color: background color of canvas.
    :return: (m + 1, c) palette of m colors to reconstruct the image (with background color prepended).
    """
    colors = RGBCMYKW.copy() if palette_type == PaletteType.RGBCMYKW else COLOR_DICT.copy()
    colors = np.concatenate((np.array([bg_color]) / 255., colors), axis=0)  # add BG color
    if palette_type == PaletteType.RGBCMYKW:
        palette = _get_palette_dithering_simulation(image, colors, n_colors=n_colors + 1, n_fixed=3)  # fix BG + white + black
    elif palette_type == PaletteType.PATCHES_SIMULATION:
        palette = _get_palette_dithering_simulation(image, colors, n_colors=n_colors + 1, n_fixed=3)  # fix BG + white + black
    elif palette_type == PaletteType.HISTOGRAM:
        palette = _get_palette_histogram(image, colors, n_colors=n_colors + 1, n_fixed=3)  # fix BG + white + black
    elif palette_type == PaletteType.HISTOGRAM_AND_SIMULATION:
        palette = _get_palette_histogram_and_dithering(image, colors, n_colors=n_colors + 1, n_fixed=3)  # fix BG + white + black
    elif palette_type == PaletteType.CLUSTERING:
        # TODO: doesn't work well, find a way to get the best palette (will probably involve some segmentation by color)
        palette = _get_palette_kmeans(image, n_colors, force_bw=False)
        palette = np.concatenate((np.array([bg_color]) / 255., palette), axis=0)  # add BG color
    else:
        raise NotImplementedError(f'palette_type {palette_type} not implemented, see PaletteType for valid values.')
    return palette


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


def _get_palette_dithering_simulation(image: np.ndarray, colors: np.ndarray, n_colors: int, n_fixed: int) -> np.ndarray:
    """
    Given an image, a palette of n colors, a number k of fixed colors and a number m of desired palette colors, choose
    the best m-k out of the n colors to add to the palette to reconstruct the image. Best here is in the sense of
    minimal dithering error. This is done by simulating dithering on small random patches from the image.
    :param image: (c, h, w) image to reconstruct using the palette (scaled to [0, 1]).
    :param colors: (n, c) color dictionary of n colors to choose from.
    :param n_colors: m desired colors in the palette (m <= n).
    :param n_fixed: k fixed colors in the palette (first k out of n, k <= m).
    :return: (m, c) palette of m colors to reconstruct the image.
    """
    n_patches = 8
    max_size = 512
    patch_size = 64
    n_palette = colors.shape[0]
    c, h, w = image.shape
    assert n_fixed <= n_colors <= colors.shape[0], (f'number of colors must satisfy n_fixed <= n_colors <= n_palette, '
                                                    f'received ({n_fixed}, {n_colors}, {n_palette}).')
    assert patch_size <= min(h, w), f'patch_size ({patch_size}) must be smaller than image dims ({(h, w)}).'

    # resize to max size if needed
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image.transpose((1, 2, 0)), (new_w, new_h)).transpose((2, 0, 1))
        c, h, w = image.shape

    # create image patches
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size
    patch_nums = np.random.choice(n_patches_h * n_patches_w, size=n_patches)
    patches = np.zeros((n_patches, c, patch_size, patch_size), dtype=image.dtype)
    blurred_patches = np.zeros((n_patches, c, patch_size, patch_size), dtype=image.dtype)
    for i in range(n_patches):
        h0 = (patch_nums[i] // n_patches_w) * patch_size
        w0 = (patch_nums[i] % n_patches_w) * patch_size
        patches[i] = image[:, h0:h0 + patch_size, w0:w0 + patch_size]
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


def _get_palette_histogram(
        image: np.ndarray,
        colors: np.ndarray,
        n_colors: int,
        n_fixed: int = 0,
        sigma: float = 1,
) -> np.ndarray:
    """
    Given an image, a palette of n colors, a number k of fixed colors and a number m of desired palette colors, choose
    the best m-k out of the n colors to add to the palette to reconstruct the image.
    This is done by calculating a smoothed histogram on a fixed color dictionary, and choosing the top k.
    :param image: (c, h, w) image to reconstruct using the palette (scaled to [0, 1]).
    :param colors: (n, c) color dictionary of n colors to choose from.
    :param n_colors: m desired colors in the palette (m <= n).
    :param n_fixed: k fixed colors in the palette (first k out of n, k <= m).
    :param sigma: smoothing sigma for the histogram.
    :return: (m, c) palette of m colors to reconstruct the image.
    """
    # save time by downsampling
    max_size = 256
    image = image.transpose((1, 2, 0))
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pixels = image.reshape(-1, 3)  # (n, 3)

    # calculate smoothed color dictionary distances
    n_colors -= n_fixed
    dists = np.linalg.norm(rgb2lab(pixels)[:, np.newaxis, :] - rgb2lab(colors)[np.newaxis, :, :], axis=2)  # (hw, n)
    weights = np.exp(-dists / (2 * (sigma ** 2)))
    norm = np.sum(weights, axis=1, keepdims=True)  # (hw, 1)
    norm[norm == 0] = 1e16
    weights = weights / norm
    hist = np.sum(weights, axis=0)  # (n,)

    # choose top m from histogram
    hist = hist[n_fixed:]
    top_k = np.argsort(hist)[-n_colors:][::-1] + n_fixed
    palette = colors[top_k]  # (m-k, 3)
    palette = np.concatenate([colors[:n_fixed], palette], axis=0)  # (m, 3)
    return palette


def _get_palette_histogram_and_dithering(
        image: np.ndarray,
        colors: np.ndarray,
        n_colors: int,
        n_fixed: int = 0,
        sigma: float = 1,
) -> np.ndarray:
    """
    Given an image, a palette of n colors, a number k of fixed colors and a number m of desired palette colors, choose
    the best m-k out of the n colors to add to the palette to reconstruct the image.
    This is done by first reducing the color dictionary to 10 colors by histogram method, then choosing the best k
    colors by dithering simulation method, thus gaining the tim efficiency of the former method and the precision
    of the latter.
    :param image: (c, h, w) image to reconstruct using the palette (scaled to [0, 1]).
    :param colors: (n, c) color dictionary of n colors to choose from.
    :param n_colors: m desired colors in the palette (m <= n).
    :param n_fixed: k fixed colors in the palette (first k out of n, k <= m).
    :param sigma: smoothing sigma for the histogram.
    :return: (m, c) palette of m colors to reconstruct the image.
    """
    n_reduce = max(10, n_colors)
    palette = _get_palette_histogram(image, colors, n_reduce, n_fixed, sigma)
    palette = _get_palette_dithering_simulation(image, palette, n_colors, n_fixed)
    return palette


def _get_color_dict(n_hues: int = 9, sat_range: int = 1, light_range: int = 2) -> np.ndarray:
    """
    Return a color dictionary of colors spreading the HSV color space.
    See https://github.com/sergeyk/rayleigh/tree/master
    """
    height = sat_range + 2 * light_range
    if n_hues == 8:
        hues = np.array([0.,  0.10,  0.15,  0.28, 0.51, 0.58, 0.77,  0.85])
    elif n_hues == 9:
        hues = np.array([0.,  0.10,  0.15,  0.28, 0.49, 0.54, 0.60, 0.7, 0.87])
    elif n_hues == 10:
        hues = np.array([0.,  0.10,  0.15,  0.28, 0.49, 0.54, 0.60, 0.66, 0.76, 0.87])
    sats = np.hstack((np.linspace(0, 1, sat_range + 2)[1:-1], [1] * (light_range + 1), [.4] * (light_range - 1)))
    lights = np.hstack(
        ([1] * (sat_range + 1), np.linspace(1, 0.2, light_range + 2)[1:-1], np.linspace(1, 0.2, light_range + 2)[1:-2]))

    hsv_colors = np.stack([
        np.tile(hues[np.newaxis, :], (height, 1)),
        np.tile(sats[:, np.newaxis], (1, len(hues))),
        np.tile(lights[:, np.newaxis], (1, len(hues))),
    ], axis=-1)
    rgb_colors = hsv2rgb(hsv_colors).reshape(-1, 3)
    grays = np.tile(np.array([0] + [1] + list(np.linspace(0, 1, height)[1:-1]))[:, np.newaxis], (1, 3))
    rgb_colors = np.vstack([grays, rgb_colors])
    return rgb_colors


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    from time import time
    from string_art.utils import save_image
    from string_art.utils.visualizations import print_with_colorbar, COLORBAR_PLACEHOLDER

    input_path = 'examples/fish.jpg'
    result_path = 'debug/fish_dithered.png'
    color_images_path = 'debug/fish_color_images.png'
    colors = np.array([[255., 255., 255.],
                       [255., 100., 0.],
                       [50., 150., 220.],
                       [0., 0., 0.]]) / 255.
    resolution = None
    palette_type = PaletteType.HISTOGRAM_AND_SIMULATION
    n_colors = 4

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    np.random.seed(42)
    tic = time()
    color_images, dithered = preprocess_image(input_path, colors=colors, palette_type=palette_type,
                                              n_colors=n_colors, resolution=resolution)
    preprocess_time = time() - tic
    print(f'Preprocess time: {preprocess_time:.2f}')
    print('Selected color palette:')
    for c in color_images:
        print_with_colorbar(f'{COLORBAR_PLACEHOLDER}', c)
    save_image(dithered, result_path)
    visualize_color_images(color_images, color_images_path)
