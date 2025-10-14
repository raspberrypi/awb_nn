from dataclasses import dataclass
from pathlib import Path
import os
import math
import numpy as np
import cv2
from tuning import Tuning
import time

IMAGE_SIZE = 32

MIN_TEMP = 2800
MAX_TEMP = 7600
TEMP_CLASSES = 10

def matvec(matrix, vector):
    """
    Fast matrix-vector multiplication using einsum.
    Equivalent to np.vectorize(np.matmul, signature='(m,n),(n)->(m)') but much faster.
    """
    return np.einsum('ij,...j->...i', matrix, vector)

def delta2sum(image: np.ndarray, gains: np.ndarray, limit: float) -> np.ndarray:
    """
    Calculate the delta2sum of an image.
    """
    total = 0
    image = image.reshape(-1, 3)
    for i in range(image.shape[0]):
        red_delta = image[i, 0] * gains[0] / (image[i, 1]) - 1
        blue_delta = image[i, 2] * gains[1] / (image[i, 1]) - 1
        delta2 = red_delta ** 2 + blue_delta ** 2
        delta2 = min(delta2, limit)
        total += delta2

    return total

def lsc(rgb: np.ndarray, tunings: Tuning, temp: float) -> np.ndarray:
    """
    Apply the LSC tables to an RGB image.
    """
    tables = tunings.get_lsc_tables(temp)
    rgb2 = np.zeros_like(rgb)
    for i in range(3):
        table = cv2.resize(tables[i], (rgb.shape[1], rgb.shape[0]))
        rgb2[:, :, i] = rgb[:, :, i] * table
    return rgb2

def gains_search(rgb: np.ndarray, tunings: Tuning, temp: float, tangent_search: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Search nearby temp for the best gains.
    """
    transverse_steps = tunings.get_transverse_search_range(temp, tangent_search)

    if transverse_steps is None:
        return None, None

    delta_limit = tunings.get_transverse_search_config()["delta_limit"]

    gain_steps = 1 / transverse_steps

    small = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    delta2sums = np.array([delta2sum(small, i, delta_limit) for i in gain_steps])
    red_blue = transverse_steps[np.argmin(delta2sums)]
    gains = np.array([1 / red_blue[0], 1 / red_blue[1]]) # type: ignore

    return gains, transverse_steps

def raw_to_rgb(
        raw: np.ndarray,
        tunings: Tuning,
        temp: float,
        gains: tuple[float, float] | None = None,
        transverse_search: bool = False,
        return_gains: bool = False,
        tangent_search: bool = False,
        smooth: bool = False
    ) -> np.ndarray | tuple[np.ndarray, tuple[float, float], np.ndarray]:
    """
    Convert a raw image to an RGB image.
    """
    rgb = raw.astype(np.float64)

    rgb = lsc(rgb, tunings, temp)

    if smooth:
        rgb = smooth_rgb(rgb)

    rgb = rgb.clip(0, 65535)

    # Apply the gains.
    if gains is None and transverse_search:
        gains, steps = gains_search(rgb, tunings, temp, tangent_search)

    if gains is None:
        red_blue = tunings.get_colour_values(temp)
        gains = np.array([1 / red_blue[0], 1 / red_blue[1]]) # type: ignore

    gains = np.array([gains[0], 1, gains[1]]) # type: ignore

    rgb = rgb * gains # type: ignore

    rgb = rgb.clip(0, 65535)

    # Apply the CCM.
    ccm = tunings.get_ccm(temp)
    rgb = matvec(ccm, rgb)

    if transverse_search and return_gains:
        return (rgb / 65535).clip(0, 1), gains[[0, 2]], steps

    return (rgb / 65535).clip(0, 1)

def smooth_rgb(rgb):
    """
    Smooth sharp changes in colour.
    """
    new_rgb = np.zeros_like(rgb)
    r_diff = rgb[..., 0] - rgb[..., 1]
    b_diff = rgb[..., 2] - rgb[..., 1]

    min_r_diff = np.zeros_like(r_diff)
    min_b_diff = np.zeros_like(b_diff)
    r_diff_padded = np.pad(r_diff, 1, mode='edge')
    b_diff_padded = np.pad(b_diff, 1, mode='edge')

    for i in range(min_r_diff.shape[0]):
        for j in range(min_r_diff.shape[1]):
            window = r_diff_padded[i:i+3, j:j+3].flatten()
            minimum = np.argmin(np.abs(window))
            min_r_diff[i, j] = window[minimum]

            window = b_diff_padded[i:i+3, j:j+3].flatten()
            minimum = np.argmin(np.abs(window))
            min_b_diff[i, j] = window[minimum]

    new_rgb[..., 1] = rgb[..., 1]
    new_rgb[..., 0] = new_rgb[..., 1] + min_r_diff
    new_rgb[..., 2] = new_rgb[..., 1] + min_b_diff

    return new_rgb

def pixel_to_rgb(pixel: np.ndarray, tunings: Tuning, temp: float, gains: tuple[float, float] | None = None) -> np.ndarray:
    """
    Converts a single pixel to RGB, skipping LSC.
    """
    pixel = pixel.astype(np.float64)
    if gains is None:
        red_blue = tunings.get_colour_values(temp)
        gains = np.array([1 / red_blue[0], 1, 1 / red_blue[1]]) # type: ignore
    else:
        gains = np.array([gains[0], 1, gains[1]]) # type: ignore
    pixel *= gains

    ccm = tunings.get_ccm(temp)
    pixel = matvec(ccm, pixel)

    return (pixel / 65535).clip(0, 1)

def normalise_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Normalise an RGB image.
    """
    green = rgb[..., 1]
    green = np.clip(green, 0.1, 1)
    new_rgb = np.zeros_like(rgb)
    new_rgb[..., 0] = rgb[..., 0] / green
    new_rgb[..., 2] = rgb[..., 2] / green
    new_rgb[..., 1] = rgb[..., 1]

    new_rgb = new_rgb.clip(0.01, 1)

    new_rgb[..., 0] = np.log(new_rgb[..., 0])
    new_rgb[..., 2] = np.log(new_rgb[..., 2])

    new_rgb = new_rgb + 1
    new_rgb = new_rgb / 2

    new_rgb[..., 1] = 0

    return new_rgb.clip(0, 1)

def apply_gamma(rgb: np.ndarray, tunings: Tuning) -> np.ndarray:
    """
    Apply the gamma curve to an RGB image.
    """
    rgb = (rgb * 65535).astype(np.uint16)
    gamma_curve = tunings.get_gamma_curve()
    gamma_lut = np.interp(range(65536), gamma_curve[0], gamma_curve[1], right=65535).astype(np.uint16)
    rgb = gamma_lut[rgb]
    return (rgb / 65535).clip(0, 1)

temp_classes = np.linspace(MIN_TEMP, MAX_TEMP, TEMP_CLASSES)

def temp_to_classes(temp: float) -> np.ndarray:
    """
    Convert a temperature to a class.
    """
    category = np.argmin(np.abs(temp_classes - temp))
    classes = np.zeros(TEMP_CLASSES)
    classes[category] = 1
    return classes

def classes_to_temp(classes: np.ndarray) -> float:
    """
    Convert a class to a temperature.
    """
    assert np.isclose(np.sum(classes), 1)

    weights = classes ** 2
    weights = weights / np.sum(weights)

    return np.sum(weights * temp_classes)

def get_temp(path: Path) -> float:
    """
    Get the temperature of an image.
    """
    return float(path.stem.split(",")[3])

def get_lux(path: Path) -> float:
    """
    Get the lux of an image.
    """
    return float(path.stem.split(",")[4])

def get_temp_estimate(path: Path) -> float:
    """
    Get the temperature estimate of an image.
    """
    return float(path.stem.split(",")[5])

def normalise_lux(lux: float) -> float:
    """
    Normalise a lux image.
    """
    return math.log(lux) / 10
