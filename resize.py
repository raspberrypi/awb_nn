from pathlib import Path
from converter import CalibratedDng
import cv2
import numpy as np
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.camdefs import Picamera2Camera
import os
import argparse

def resize(input_path: Path, output_dir: Path, size: int) -> None:
    try:
        dng = CalibratedDng(input_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    new_raw = np.zeros((size * 2, size * 2), dtype=np.uint16)

    for offset in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        new_raw[offset[0]::2, offset[1]::2] = cv2.resize(dng.raw_array[offset[0]::2, offset[1]::2], (size, size), interpolation=cv2.INTER_AREA)

    tags = DNGTags()
    tags.set(Tag.ImageWidth, size * 2)
    tags.set(Tag.ImageLength, size * 2)
    tags.set(Tag.BitsPerSample, dng.exif_data["Exif.Image.BitsPerSample"].value)
    tags.set(Tag.BlackLevel, dng.exif_data["Exif.Image.BlackLevel"].value)
    tags.set(Tag.BlackLevelRepeatDim, dng.exif_data["Exif.Image.BlackLevelRepeatDim"].value)
    tags.set(Tag.WhiteLevel, dng.exif_data["Exif.Image.WhiteLevel"].value)
    ccm = dng.exif_data["Exif.Image.ColorMatrix1"].value
    ccm = list(map(lambda x: x.as_integer_ratio(), ccm))
    tags.set(Tag.ColorMatrix1, ccm)
    tags.set(Tag.CameraCalibration1, [(1, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (1, 1)])
    tags.set(Tag.SamplesPerPixel, dng.exif_data["Exif.Image.SamplesPerPixel"].value)
    tags.set(Tag.PhotometricInterpretation, dng.exif_data["Exif.Image.PhotometricInterpretation"].value)
    tags.set(Tag.CFARepeatPatternDim, dng.exif_data["Exif.Image.CFARepeatPatternDim"].value)
    cfa_pattern = dng.exif_data["Exif.Image.CFAPattern"].value
    cfa_pattern = list(map(int, cfa_pattern.split(" ")))
    tags.set(Tag.CFAPattern, cfa_pattern)

    exposure_time = dng.exif_data["Exif.Image.ExposureTime"].value
    tags.set(Tag.ExposureTime, [exposure_time.as_integer_ratio()])
    tags.set(Tag.PhotographicSensitivity, dng.exif_data["Exif.Image.ISOSpeedRatings"].value)

    if dng.aperture is not None:
        tags.set(Tag.FNumber, dng.aperture)

    white_balance = dng.exif_data["Exif.Image.AsShotNeutral"].value
    white_balance = list(map(lambda x: x.as_integer_ratio(), white_balance))
    tags.set(Tag.AsShotNeutral, white_balance)

    tags.set(Tag.CalibrationIlluminant1, dng.exif_data["Exif.Image.CalibrationIlluminant1"].value)
    baseline_exposure = dng.exif_data["Exif.Image.BaselineExposure"].value
    baseline_exposure = baseline_exposure.as_integer_ratio()
    tags.set(Tag.BaselineExposure, [baseline_exposure])

    dng_writer = RAW2DNG()
    dng_writer.options(tags, path="", compress=False)

    filename = input_path.stem.split(",")
    filename = filename[0:3]
    dng.do_lsc(dng.calculate_colour_temp())
    gains = dng.gains()
    filename = filename + [str(gains[0]), str(gains[1])]
    filename = ",".join(filename)

    dng_writer.convert(new_raw, str(output_dir / f"{filename}.dng"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize DNG files")
    parser.add_argument("source_dir", type=str)
    parser.add_argument("destination_dir", type=str)
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()

    for (dirpath, dirnames, filenames) in os.walk(args.source_dir):
        os.makedirs(dirpath.replace(args.source_dir, args.destination_dir), exist_ok=True)
        for filename in filenames:
            if filename.endswith(".dng"):
                resize(Path(dirpath) / filename, Path(dirpath.replace(args.source_dir, args.destination_dir)), args.size)
                print(f"Resized {filename}")
