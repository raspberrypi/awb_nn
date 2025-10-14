
#! /usr/bin/env python3

import rawpy
import numpy as np
import os
from pathlib import Path
import cv2
import argparse
import sys
from tuning import Tuning
from rgb import save as save_rgb

class Dng:
    """
    A class to represent a DNG file.

    Defective pixel correction (DPC), lens shading correction (LSC), digital gain and denoise,
    using the supplied NAFNET models, can be applied directly to the raw image.

    The raw image can be saved to a new DNG file, and/or converted to an RGB image.
    """

    def __init__(self, dng_filename: str, target: str, tuning = None, sensor = None) -> None:
        """
        Initialize Dng object by loading a DNG file. Optionally, a Tuning object can be provided, otherwise it
        will be loaded from the DNG filename.

        Args:
            filename (str): Path to the DNG file
            tuning (Tuning): Tuning object for this camera
            sensor (str): Sensor model name, if not provided, it will be read from the DNG file
        """
        self.dng_filename = Path(dng_filename)
        if not self.dng_filename.exists():
            raise FileNotFoundError(f"DNG file {dng_filename} not found")

        self.raw = rawpy.imread(str(self.dng_filename))

        self.model = sensor if sensor else self.exif_data["Exif.Image.Model"].value
        # Some Picamera2 DNG files won't have a valid sensor recorded.
        if not sensor and self.model == "PiDNG / PiCamera2":
            raise ValueError("Sensor model not found in the DNG file - try the -s option to specify it")

        # Each pair holds the (x, y) offsets for the R, Gr, B and Gb channels respectively
        self.raw_offsets = [None, None, None, None]
        self.raw_offsets[self.raw.raw_pattern[0, 0]] = (0, 0)
        self.raw_offsets[self.raw.raw_pattern[0, 1]] = (0, 1)
        self.raw_offsets[self.raw.raw_pattern[1, 0]] = (1, 0)
        self.raw_offsets[self.raw.raw_pattern[1, 1]] = (1, 1)

        if tuning:
            self.tuning = tuning
        else:
            tuning_file = Tuning.find(self.model, target.lower())
            self.tuning = Tuning.load(tuning_file)

        self.raw_array_backup = self.raw_array.copy()

    @property
    def raw_array(self) -> np.ndarray:
        """
        Return the raw Bayer image.
        """
        return self.raw.raw_image_visible

    @property
    def black_level(self) -> int:
        """
        Return the black level for the image (assume same for all channels).
        """
        return self.raw.black_level_per_channel[0]

    @property
    def white_level(self) -> int:
        """
        Return the white level for the image.
        """
        return self.raw.white_level

    @property
    def camera_white_balance(self) -> np.ndarray:
        """
        Return the camera white balance for the image.
        """
        return self.raw.camera_whitebalance

    def rgb_averages(self, x0, y0, x1, y1):
        """
        Return the R, G, B averages of the pixels in the rectangle defined by (x0, y0) and (x1, y1).

        Args:
            x0 (int): Left coordinate of the rectangle
            y0 (int): Top coordinate of the rectangle
            x1 (int): Right coordinate of the rectangle
            y1 (int): Bottom coordinate of the rectangle

        Returns:
            list(float): Average of the pixels in the rectangle ordered as (R, G, B)
        """
        x0 = int(x0) & ~1
        y0 = int(y0) & ~1
        R_ave = np.mean(self.raw_array[y0 + self.raw_offsets[0][1]:y1:2, x0 + self.raw_offsets[0][0]:x1:2])
        Gr_ave = np.mean(self.raw_array[y0 + self.raw_offsets[1][1]:y1:2, x0 + self.raw_offsets[1][0]:x1:2])
        B_ave = np.mean(self.raw_array[y0 + self.raw_offsets[2][1]:y1:2, x0 + self.raw_offsets[2][0]:x1:2])
        Gb_ave = np.mean(self.raw_array[y0 + self.raw_offsets[3][1]:y1:2, x0 + self.raw_offsets[3][0]:x1:2])
        G_ave = (Gr_ave + Gb_ave) / 2
        R_ave -= self.black_level
        G_ave -= self.black_level
        B_ave -= self.black_level
        return [R_ave, G_ave, B_ave]

    def close(self):
        """
        Close the DNG file to free resources.
        """
        self.raw.close()
        self.raw = None
        self.raw_offsets = None
        self.tuning = None
        self.model = None
        self.dng_filename = None

    def __del__(self):
        self.close()

    def restore(self):
        """
        Restore the raw image to the state it was in when the object was created.
        """
        self.raw_array[...] = self.raw_array_backup

    def do_dpc(self, extra: float = 0.25) -> None:
        """
        Apply simple DPC (Defective Pixel Correction) to the raw image. This alters the raw image in place,
        so it should not really be called more than once. It should work adequately for single pixel defects.

        Args:
            extra (float): Allow slightly wider limits for pixel clipping (default is 0.25)
        """
        # We're going to ignore the two edge rows/columns, unless we see a need later.
        arrays = [
            self.raw_array[:-4, :-4],
            self.raw_array[:-4, 2:-2],
            self.raw_array[:-4, 4:],
            self.raw_array[2:-2, :-4],
            self.raw_array[2:-2, 4:],
            self.raw_array[4:, :-4],
            self.raw_array[4:, 2:-2],
            self.raw_array[4:, 4:],
        ]
        max_array = np.max(arrays, axis=0)
        min_array = np.min(arrays, axis=0)
        centre = self.raw_array[2:-2, 2:-2]
        # Clip central pixel to the min/max of the neighbours, plus a little "extra".
        max_array = max_array.astype(np.float32)
        min_array = min_array.astype(np.float32)
        diff = (max_array - min_array) * extra
        max_array += diff
        min_array -= diff
        centre = centre.astype(np.float32).clip(min_array, max_array)

        self.raw_array[2:-2, 2:-2] = centre.clip(0, self.white_level).astype(np.uint16)

    def estimate_colour_temp(self, colour_gains=None) -> float:
        """
        Estimate the colour temperature of the image using the tuning file, using the camera
        white balance if no colour gains are provided.

        Returns:
            float: Estimated colour temperature in Kelvin
        """
        if colour_gains is None:
            colour_values = 1.0 / np.array(self.camera_white_balance)[[0, 2]]
        else:
            colour_values = 1.0 / np.array(colour_gains)
        colour_temp = self.tuning.get_colour_temp(colour_values)
        return colour_temp

    def do_lsc(self, colour_temp: float = None) -> None:
        """
        Apply LSC (Lens Shading Correction) to the raw image. This alters the raw image in place,
        so it should not really be called more than once.

        Args:
            colour_temp (float): Colour temperature to use for LSC. If None, it will be estimated
                from the camera white balance and the tuning file.
        """
        raw_image = self.raw_array.astype(np.float32)

        # Subtract the black level.
        raw_image -= self.black_level
        raw_image = raw_image.clip(0, self.white_level)

        # Get the lens shading correction tables. First, we need to estimate the colour temperature.
        if colour_temp is None:
            colour_temp = self.estimate_colour_temp()
        r_table, g_table, b_table = self.tuning.get_lsc_tables(colour_temp)

        # Apply the lens shading correction.
        w, h = raw_image.shape[1::-1]
        half_res = (w // 2, h // 2)
        lsc_tables = [r_table, g_table, b_table, g_table]
        for component in range(4):
            offsets = self.raw_offsets[component][0], self.raw_offsets[component][1]
            raw_image[offsets[0]::2, offsets[1]::2] *= cv2.resize(lsc_tables[component], half_res)

        self.raw_array[...] = (raw_image + self.black_level).clip(0, self.white_level).astype(np.uint16)

    def do_digital_gain(self, digital_gain: float) -> None:
        """
        Apply digital gain to the raw image. This alters the raw image in place,
        so it should not really be called more than once.
        """
        array = self.raw_array.astype(np.float32) - self.black_level
        array *= digital_gain
        array += self.black_level
        self.raw_array[...] = array.clip(0, self.white_level).astype(np.uint16)

    def convert(self, colour_gains=None, gamma=None, median_filter_passes=1, output_bps=8) -> np.ndarray:
        """
        Convert the raw image to an RGB image using rawpy. You should consider whether you want
        to apply denoise, DPC or LSC before calling this.

        Args:
            colour_gains (tuple): If None, the camera white balance will be used. Otherwise, pass a pair of
                numbers defining the red and blue gains.
            gamma (tuple): If None, the gamma curve from the tuning file will be used. Otherwise, pass a pair of
                numbers defining a gamma curve in the manner or rawpy.
            median_filter_passes (int): Number of median filter passes to apply.
            output_bps (int): Output bit depth (8 or 16 bits only).

        Returns:
            np.ndarray: RGB image
        """
        use_camera_wb = True
        user_wb = None
        if colour_gains is not None:
            use_camera_wb = False
            red, blue = colour_gains
            user_wb = [red, 1.0, blue, 1.0]
            min_gain = min(red, blue)
            user_wb = (np.array(user_wb) / min_gain).tolist()

        rgb_image = self.raw.postprocess(
            use_camera_wb=use_camera_wb,
            user_wb=user_wb,
            no_auto_bright=True,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.DCB,
            median_filter_passes=median_filter_passes,
            gamma=(1.0, 1.0) if gamma is None else gamma,
            output_bps=16)

        # If no gamma was supplied, use the one from the tuning file.
        if gamma is None:
            gamma_curve = self.tuning.get_gamma_curve()
            gamma_lut = np.interp(range(65536), gamma_curve[0], gamma_curve[1], right=65535).astype(np.uint16)
            rgb_image[...] = gamma_lut[rgb_image]

        if output_bps == 16:
            pass  # should be 16 bit already
        elif output_bps == 8:
            rgb_image = (rgb_image >> 8).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported output bit depth: {output_bps}")
        return rgb_image

# Helper function to parse string "num1,num2" to tuple of floats
def parse_two_values(num_str: str) -> tuple[float, float]:
    try:
        parts = num_str.split(',')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("Must be two comma-separated numbers (e.g. '2.2,4.5')")
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        raise argparse.ArgumentTypeError("Values must be numbers (e.g. '2.2,4.5')")

if __name__ == "__main__":
    """
    Command line interface to process a DNG file. The tool can be used to:

    * Apply any or all of DPC, LSC, digital gain,
      to the raw image data, which can be saved to a new DNG file.
    * Additionaly, the raw data, after any processing, can be converted to an RGB image and
      saved to another file.

    Usage:

    python dng.py --input input.dng --output-rgb output.jpg
      Simple example where input.dng is converted to an RGB image and saved to output.jpg.
      LSC is applied by default, but there is no DPC, denoise or digital gain.

    Notes:
    * When creating an RGB output, the --colour-gains and --gamma options can be used to adjust the
      output image.
    * At the time of writing, DNG files from Picamera2 don't record the sensor model, so the -s option
      should be used to specify it. The problem is being fixed in Picamera2 and PiDNG.

    Type "python dng.py --help" for more options.
    """

    parser = argparse.ArgumentParser(description="Process a DNG file.")
    parser.add_argument("-i", "--input", required=True, help="Input DNG filename")
    parser.add_argument("-s", "--sensor", help="Sensor model name (e.g. imx477, optional though some DNG files may need it)")
    parser.add_argument("--overlap", type=int, default=16, help="Number of overlap pixels between image patches")
    parser.add_argument("--tuning", help="Tuning filename (optional)")
    parser.add_argument("--dpc", choices=["on", "off"], default="off", help="Enable or disable DPC (Defective Pixel Correction) (default: off)")
    parser.add_argument("--lsc", choices=["on", "off"], default="on", help="Enable or disable LSC (Lens Shading Correction) (default: on)")
    parser.add_argument("--colour-gains", type=parse_two_values, default=None, help="Red and blue gains as 'num1,num2' (e.g. '1.7,2.3'). Defaults to DNG file gains.")
    parser.add_argument("--digital-gain", type=float, default=1.0, help="Apply digital gain (default: 1.0)")
    parser.add_argument("--gamma", type=parse_two_values, default=None, help="Gamma curve as 'num1,num2' (e.g. '2.2,4.5') as per rawpy. Defaults to tuning file gamma.")
    parser.add_argument("--output-rgb", help="Output RGB filename (optional)")
    parser.add_argument("-y", "--yes", action="store_true", help="Overwrite existing output files (default: False)")
    args = parser.parse_args()

    tuning = None
    if args.tuning:
        tuning = Tuning.load(args.tuning)

    dng = Dng(args.input, sensor=args.sensor, tuning=tuning)

    if args.dpc == "on":
        dng.do_dpc()

    if args.lsc == "on":
        dng.do_lsc()

    if args.digital_gain != 1.0:
        dng.do_digital_gain(args.digital_gain)

    if args.output_rgb:
        rgb = dng.convert(colour_gains=args.colour_gains, gamma=args.gamma, median_filter_passes=1, output_bps=8)
        save_rgb(args.output_rgb, rgb, overwrite=args.yes)
