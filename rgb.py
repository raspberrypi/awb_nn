#! /usr/bin/env python3

import cv2
import numpy as np
import os
import argparse
from pathlib import Path

def load(filename: str) -> np.ndarray:
    """
    Load an RGB image from a file.

    Args:
        filename (str): Path to the image file

    Returns:
        np.ndarray: RGB image array
    """
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save(filename: str, image: np.ndarray, overwrite: bool = False) -> None:
    """
    Save an RGB image to a file. You can save a 16-bit PNG, but for JPEG it will be saved as 8-bit.

    Args:
        filename (str): Path to save the image
        image (np.ndarray): RGB image array to save (8 or 16-bit)
        overwrite (bool): Whether to overwrite the output file if it already exists
    """
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"File {filename} already exists")

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR, not RGB
    if filename.endswith(".jpg") and image.dtype == np.uint16:
        bgr_image = (bgr_image >> 8).astype(np.uint8)
    cv2.imwrite(filename, bgr_image)