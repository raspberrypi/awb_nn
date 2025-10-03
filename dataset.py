from pathlib import Path
import os
import tensorflow as tf
from typing import Optional
from data import get_temp, temp_classes, TEMP_CLASSES, MIN_TEMP, MAX_TEMP

temp_classes_tensor = tf.constant(temp_classes, dtype=tf.float32)

def temp_to_classes_tensor(temp):
    """
    Convert a temperature to a class.
    """
    temp = tf.convert_to_tensor(temp, dtype=tf.float32)
    diff = tf.abs(temp_classes_tensor - temp)
    category = tf.argmin(diff)
    classes = tf.one_hot(category, depth=TEMP_CLASSES, dtype=tf.float32)

    return classes

def create_filename_dataset(dng_folder: Path, duplicates: dict):
    """
    Create a dataset containing the filenames of the images.
    """
    paths = []
    for (dirpath, dirnames, filenames) in os.walk(dng_folder):
        for filename in filenames:
            if filename.endswith(".png"):
                new_path = Path(dirpath) / filename
                new_filename = str(new_path)
                new_relative_path = new_path.relative_to(dng_folder)
                new_relative_filename = str(new_relative_path)
                # See if we need to repeat this training example. There would be in entry in
                # the duplicates table. Select the one with the longest match.
                matches = [k for k in duplicates if new_relative_filename.strip().startswith(k)]
                repeats = duplicates[max(matches, key=len)] if matches else 1
                new_paths = [new_filename] * repeats
                paths += new_paths

    if len(paths) == 0:
        print("WARNING: no test images")
        return None

    return tf.data.Dataset.from_tensor_slices(paths)

def filename_to_pair(filename: str, image_shape: tuple[int, int] = (32, 32)):
    """
    Convert a filename to an input and output pair.
    """
    data = tf.io.read_file(filename)
    image = tf.image.decode_png(data, channels=3, dtype=tf.uint16)
    image = tf.image.resize(image, image_shape)
    image = image / 65535
    image = tf.reshape(image, (image_shape[0], image_shape[1], 3))

    temp = tf.strings.split(filename, ",")[3]
    temp = tf.strings.to_number(temp, out_type=tf.float32)
    lux = tf.strings.split(filename, ",")[4]
    lux = tf.strings.to_number(lux, out_type=tf.float32)

    return (image, lux), temp

def create_dataset(dng_folder: Path, image_shape: tuple[int, int], batch_size: int, duplicates: dict):
    """
    Create a dataset containing the images and their temperatures.
    """
    ds = create_filename_dataset(dng_folder, duplicates)
    if ds is None:
        return ds
    ds = ds.map(lambda x: filename_to_pair(x, image_shape))
    ds = ds.batch(batch_size)
    ds = ds.cache()
    ds = ds.shuffle(1000)
    return ds

def read_duplicate_filenames(duplicate_file: Optional[Path]):
    duplicates = {}
    if not duplicate_file:
        return duplicates
    if not os.path.isfile(duplicate_file):
        print("Duplicates file", duplicate_file, "not found - continuing without")
        return duplicates

    with open(duplicate_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            split_line = line.split()
            if len(split_line) < 2:
                continue
            filename, count = split_line
            if filename in duplicates:
                duplicates[filename] += int(count)
            else:
                duplicates[filename] = int(count)

    return duplicates

def create_dataset_pair(parent_folder: Path, image_shape: tuple[int, int], batch_size: int = 16, duplicate_file: Optional[Path] = None):
    """
    Create a pair of datasets, one for training and one for validation.
    """
    duplicates = read_duplicate_filenames(duplicate_file)
    train_ds = create_dataset(parent_folder / "train", image_shape, batch_size, duplicates)
    val_ds = create_dataset(parent_folder / "test", image_shape, batch_size, {})
    return train_ds, val_ds
