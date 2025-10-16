import os
from pathlib import Path
import rawpy
from tuning import Tuning
import numpy as np
import cv2
from rgb import save as save_rgb
import random
from data import raw_to_rgb, apply_gamma
import argparse
import shutil
import json
import subprocess

# pyexiv2 causes problems when python is built from source
USE_EXIFTOOL = False

if USE_EXIFTOOL:
    if shutil.which("exiftool") is None:
            raise ValueError("exiftool is not installed - install it with 'sudo apt install libimage-exiftool-perl'")
else:
    import pyexiv2


class CalibratedDng:
    def __init__(self, dng_filename: str, target: str, tuning = None, sensor = None) -> None:
        """
        Initialize Dng object by loading a DNG file. Optionally, a Tuning object can be provided, otherwise it
        will be loaded from the DNG filename.
        The Dng filename must be in the format: <user_id>,<sensor>,<id>,<x1>,<y1>,<x2>,<y2>.dng
        or <user_id>,<sensor>,<id>,<red gain>,<blue gain>.dng
        The gains are then read from the filename or calculated by assuming the rectangle specified is grey.

        Args:
            filename (str): Path to the DNG file
            tuning (Tuning): Tuning object for this camera
            sensor (str): Sensor model name, if not provided, it will be read from the DNG file
        """
        self.dng_filename = Path(dng_filename)

        if len(self.dng_filename.stem.split(",")) not in [5, 7]:
            raise ValueError("DNG file name must be in the format: \
                    <user_id>,<sensor>,<id>,<x1>,<y1>,<x2>,<y2>.dng \
                    or <user_id>,<sensor>,<id>,<red gain>,<blue gain>.dng")

        if not self.dng_filename.exists():
            raise FileNotFoundError(f"DNG file {dng_filename} not found")

        self.raw = rawpy.imread(str(self.dng_filename))

        # rawpy doesn't read all the exif tags, so read the missing ones with pyexiv2 or exiftool
        if USE_EXIFTOOL:
            self.exif_data = json.loads(subprocess.check_output(["exiftool", "-j", str(self.dng_filename)]))
            self.exif_data = self.exif_data[0]

        else:
            self.exif_data = pyexiv2.ImageMetadata(str(self.dng_filename))
            self.exif_data.read()
        self.model = self.dng_filename.stem.split(",")[1]

        # Each pair holds the (x, y) offsets for the R, Gr, B and Gb channels respectively
        self.raw_offsets : list[tuple[int, int]] = [None, None, None, None]  # type: ignore
        self.raw_offsets[self.raw.raw_pattern[0, 0]] = (0, 0)
        self.raw_offsets[self.raw.raw_pattern[0, 1]] = (0, 1)
        self.raw_offsets[self.raw.raw_pattern[1, 0]] = (1, 0)
        self.raw_offsets[self.raw.raw_pattern[1, 1]] = (1, 1)

        if tuning:
            self.tuning = tuning
        else:
            tuning_file = Tuning.find(self.model, target.lower())
            self.tuning = Tuning.load(tuning_file)
            print("Loaded tuning file", tuning_file)

        if len(self.dng_filename.stem.split(",")) == 7:
            _, _, _, x1, y1, x2, y2 = self.dng_filename.stem.split(",")
            self.red_gain = None
            self.blue_gain = None
            self.gray_area = (even(int(x1)), even(int(y1)), even(int(x2)), even(int(y2)))
        elif len(self.dng_filename.stem.split(",")) == 5:
            _, _, _, red_gain, blue_gain = self.dng_filename.stem.split(",")
            self.red_gain = float(red_gain)
            self.blue_gain = float(blue_gain)
            self.gray_area = None
        else:
            raise ValueError("Should never happen")

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

    @property
    def aperture(self) -> float:
        """
        Return the aperture of the image.
        """
        if USE_EXIFTOOL:
            return None # Raspberry pi cameras don't record the aperture
        else:
            return self.exif_data.get_aperture()

    @property
    def shutter_speed(self) -> float:
        """
        Return the shutter speed of the image.
        """
        if USE_EXIFTOOL:
            exposure_time = self.exif_data.get("ExposureTime")
            if exposure_time is None:
                return None
            a, b = exposure_time.split("/")
            return 1000000 * int(a) / int(b)
        else:
            return 1000000 * self.exif_data["Exif.Image.ExposureTime"].value

    @property
    def gain(self) -> float:
        """
        Return the gain of the image.
        """
        if USE_EXIFTOOL:
            iso_speed_ratings = self.exif_data.get("ISOSpeedRatings")
            if iso_speed_ratings is None:
                return None
            return iso_speed_ratings / 100
        else:
            return self.exif_data["Exif.Image.ISOSpeedRatings"].value / 100

    def estimate_colour_temp(self) -> float:
        """
        Estimate the colour temperature of the image, using the camera white balance and the tuning file.

        Returns:
            float: Estimated colour temperature in Kelvin
        """
        red_blue = 1.0 / np.array(self.camera_white_balance)[[0, 2]]
        colour_temp = self.tuning.get_colour_temp(red_blue)
        return colour_temp

    def estimate_gains(self) -> tuple[float, float]:
        """
        Return the red and blue gains estimated by the camera.
        """
        return (self.camera_white_balance[0], self.camera_white_balance[2])

    def calculate_colour_temp(self) -> float:
        """
        Calculate the colour temperature of the image, using the gray area or provided gains.
        """
        return self.tuning.get_colour_temp(self.red_blue())

    def do_lsc(self, colour_temp: float | None = None) -> None:
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
            table = cv2.resize(lsc_tables[component], half_res)
            raw_image[offsets[0]::2, offsets[1]::2] *= cv2.resize(table, half_res)

        # Don't add back the black level here, it just loses us a bit of headroom for the LSC gains,
        # and we don't do it in the hardware.
        self.raw_array[...] = raw_image.clip(0, self.white_level).astype(np.uint16)

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
        if colour_gains:
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

    def average_gray_area(self) -> tuple[float, float, float]:
        """
        Average the colour values in the gray area of the image.
        """
        averages = np.zeros(4)
        for component in range(4):
            offsets = self.raw_offsets[component][0], self.raw_offsets[component][1]
            area = self.raw_array[self.gray_area[1]+offsets[0]:self.gray_area[3]+offsets[0]:2,
                                  self.gray_area[0]+offsets[1]:self.gray_area[2]+offsets[1]:2]
            average = np.median(area)
            averages[component] = average

        averages -= self.black_level
        averages /= self.white_level
        return (averages[0], (averages[1] + averages[3]) / 2, averages[2])

    def gains(self) -> tuple[float, float]:
        """
        Return the red and blue gains for the image from the gray area or provided gains.
        """
        if self.red_gain is None or self.blue_gain is None:
            averages = self.average_gray_area()
            return (averages[1] / averages[0], averages[1] / averages[2])
        else:
            return (self.red_gain, self.blue_gain)

    def red_blue(self) -> tuple[float, float]:
        """
        Return the inverse of the red and blue gains from the gray area or provided gains.
        """
        gains = self.gains()
        return (1.0 / gains[0], 1.0 / gains[1])


def even(x: int) -> int:
    """
    Rounds down to the nearest even number.
    """
    return x // 2 * 2

def save(path: Path, rgb: np.ndarray, overwrite: bool = False) -> None:
    """
    Save an RGB image to a file.

    Args:
        path (Path): The path to save the image to.
        rgb (ndarray): The RGB image to save.
        overwrite (bool): Whether to overwrite the file if it already exists.
    """
    save_rgb(str(path), rgb, overwrite=overwrite)

def estimate_lux(rgb: np.ndarray, dng: CalibratedDng) -> float:
    """
    Estimate the lux of a DNG.

    Args:
        rgb (ndarray): The RGB image to estimate the lux of.
        dng (Dng): The DNG object to get the camera tuning and image metadata from.

    Returns:
        float: The estimated lux.
    """
    colour_gains = 1 / dng.red_blue()[0], 1 / dng.red_blue()[1]
    Y = np.mean(rgb * np.array([0.299 * colour_gains[0], 0.587, 0.114 * colour_gains[1]]) * 3)
    lux = dng.tuning.calculate_lux(Y, dng.gain, dng.aperture, dng.shutter_speed)
    # This lux estimate seems to be across the board some 5-10% lower than the hardware sees.
    lux *= 1.05
    return lux

def dng_to_rgb(dng: CalibratedDng, black_level: bool = True) -> np.ndarray:
    """
    Convert a DNG to an RGB by stacking the channels and applying the black and white levels.

    Args:
        dng (Dng): The DNG object to convert.

    Returns:
        ndarray: The RGB image in the range 0-65535.
    """
    channels = [dng.raw_array[offset[0]::2, offset[1]::2].astype(np.float64) for offset in dng.raw_offsets]
    channels[1] = (channels[1] + channels[3]) / 2
    rgb = np.stack(channels[:3], axis=2)
    if black_level:
        rgb -= dng.black_level
    rgb = rgb / dng.white_level
    rgb = rgb.clip(0, 1)
    return (rgb * 65535).astype(np.uint16)


def rgb_to_input(rgb: np.ndarray, tunings: Tuning, lux: float) -> np.ndarray:
    """
    Convert an RGB image to an input image for the neural network.

    Args:
        rgb (ndarray): The RGB image to convert.
        tunings (Tuning): The tuning object to use for the conversion.
        lux (float): The lux of the image.

    Returns:
        ndarray: The input for the neural network.
    """
    rgb = raw_to_rgb(rgb, tunings, 5000)
    #rgb = normalise_rgb(rgb)
    #rgb[..., 1] = normalise_lux(lux)
    return rgb


def process_folder(
        dng_folder: Path,
        output_folder: Path,
        target: str,
        resize: tuple[int, int] | None = None,
        split: float | None = None,
        colour_test: bool = True,
        train_folder: Path | None = None,
        test_folder: Path | None = None) -> None:
    """
    Process a folder of DNG files to create a training and test set.

    Args:
        dng_folder (Path): The folder containing the DNG files.
        output_folder (Path): The folder to save the processed images.
        resize (tuple[int, int] | None): The size to resize the images to.
        split (float | None): The proportion of image to use for testing (0-1).
            If not specified, all images will be saved to the output folder.
        colour_test (bool): Output images with the correct gains applied to check they are correct.
        train_folder (Path | None): The folder to save the training images. If not specified, "train" will be used.
        test_folder (Path | None): The folder to save the test images. If not specified, "test" will be used.
    """
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    if split is not None:
        random.seed(42)
        if train_folder is None:
            train_folder = Path("train")
        if test_folder is None:
            test_folder = Path("test")
        train_folder = output_folder / train_folder
        test_folder = output_folder / test_folder
        if not train_folder.exists():
            train_folder.mkdir(parents=True)
        if not test_folder.exists():
            test_folder.mkdir(parents=True)

    for (dirpath, dirnames, filenames) in os.walk(dng_folder):
        for filename in filenames:
            if filename.endswith(".dng"):
                try:
                    dng = CalibratedDng(str(Path(dirpath) / filename), target)
                    temp = dng.calculate_colour_temp()
                    # Apply LSC before estimating lux on VC4 platforms
                    if target == "VC4":
                        backup = dng.raw_array.copy()
                        dng.do_lsc(temp)
                        rgb = dng_to_rgb(dng, black_level=False)
                        lux = estimate_lux(rgb, dng)
                        # But we'll undo the LSC as raw_to_rgb will re-apply it for the correct
                        # colour temp (5000K)
                        dng.raw_array[...] = backup
                        rgb = dng_to_rgb(dng, black_level=True)
                    else:
                        rgb = dng_to_rgb(dng, black_level=True)
                        lux = estimate_lux(rgb, dng)
                    if resize is not None:
                        rgb = cv2.resize(rgb, resize, interpolation=cv2.INTER_AREA)
                        
                    if colour_test:
                        rgb = raw_to_rgb(rgb, dng.tuning, temp, gains=dng.gains())
                        rgb = apply_gamma(rgb, dng.tuning)
                        rgb = (rgb * 65535).astype(np.uint16)
                    else:
                        rgb = rgb_to_input(rgb, dng.tuning, lux)
                        rgb = (rgb * 65535).astype(np.uint16)

                    new_filename = filename.removesuffix(".dng").split(",")[0:3]
                    new_filename = new_filename + [str(round(temp)),
                                                   str(round(lux)),
                                                   str(round(dng.estimate_colour_temp()))]
                    new_filename = ",".join(new_filename) + ".png"

                    if split is not None and random.random() < split:
                        output_path = test_folder / new_filename
                    elif split is not None:
                        output_path = train_folder / new_filename
                    else:
                        output_path = output_folder / new_filename

                    check = ((rgb.astype(np.float32) * 1000 / 65535).astype(np.int32)).astype(np.float32) / 1000

                    save(output_path, rgb, overwrite=True)
                    print(f"Saved {output_path}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    raise e
                    continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "input_dir",
        type=Path,
        help="The folder containing the DNG files",
        metavar="input-dir"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="The folder to save the processed images",
        metavar="output-dir"
    )
    parser.add_argument(
        "-t", "--target",
        type=str,
        help="Target platform, either PISP or VC4",
        required=True
        )
    parser.add_argument(
        "--resize",
        type=str,
        help="The size to resize the images to"
    )
    parser.add_argument(
        "--colour-test",
        action="store_true",
        help="Output images with the correct gains applied to check they are correct"
    )
    parser.add_argument("--split",
        type=float,
        default=0,
        help="The proportion of image to use for testing (0-1). \
            If not specified, all images will be saved to the output folder."
    )

    args = parser.parse_args()

    target = args.target.upper()
    if target not in ("VC4", "PISP"):
        raise ValueError(f"Target {args.target} not recognised - use one of VC4 or PISP")

    if args.resize is None:
        resize = (16, 12) if target == "VC4" else (32, 32)
        print("Chosen size", resize, "for platform", target)
    else:
        resize = tuple(int(x) for x in args.resize.split(","))
        if len(resize) != 2:
            raise argparse.ArgumentError("Resize must be a pair of integers in the format WIDTH,HEIGHT")

    if args.split is not None and (args.split < 0 or args.split > 1):
        raise argparse.ArgumentError("Split must be between 0 and 1")

    process_folder(args.input_dir, args.output_dir, target, resize, args.split, args.colour_test)
