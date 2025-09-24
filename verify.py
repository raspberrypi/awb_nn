import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from converter import CalibratedDng, dng_to_rgb, rgb_to_input, estimate_lux
import cv2
import glob
from data import raw_to_rgb, apply_gamma
import tensorflow as tf
from train import MiredLoss, ClassesToTemp, PreprocessImage

def predict(image, model, tuning, lux):
    image_shape = model.input_shape[0][1:3]
    input_image = cv2.resize(image, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_AREA)
    input_image = rgb_to_input(input_image, tuning, lux).astype(np.float32)
    input_image = np.expand_dims(input_image, axis=0)
    input_lux = np.expand_dims(lux, axis=0)
    temp = model.predict([input_image, input_lux])[0]
    gains = raw_to_rgb(image, tuning, temp, transverse_search=True, return_gains=True, tangent_search=False)[1]
    return temp, gains

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="The image to test as a DNG file", required=True)
    parser.add_argument("--model", type=Path, help="The model to test. Must be a keras model.", required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, custom_objects={
        "MiredLoss": MiredLoss,
        "ClassesToTemp": ClassesToTemp,
        "PreprocessImage": PreprocessImage
    })

    filenames = [args.image]
    for filename in filenames:
        dng = CalibratedDng(filename)
        image = dng_to_rgb(dng)
        dng.do_lsc()
        lux = estimate_lux(image, dng)
        real_gains = dng.gains()
        real_temp = dng.tuning.get_colour_temp((1 / real_gains[0], 1 / real_gains[1]))

        camera_temp = dng.estimate_colour_temp()
        camera_gains = (dng.camera_white_balance[0], dng.camera_white_balance[2])

        model_temp, model_gains = predict(image, model, dng.tuning, lux)

        images = [
            raw_to_rgb(image, dng.tuning, real_temp, real_gains),
            raw_to_rgb(image, dng.tuning, camera_temp, camera_gains),
            raw_to_rgb(image, dng.tuning, model_temp, model_gains),
        ]

        images = list(map(lambda img: apply_gamma(img, dng.tuning), images))

        temps = [real_temp, camera_temp, model_temp]
        gains = [real_gains, camera_gains, model_gains]
        labels = ["Real", "Camera", "Model"]

        for i, image in enumerate(images):
            plt.subplot(1, 3, i + 1)
            plt.imshow(image)
            plt.title(f"{labels[i]}: {temps[i]:.0f}K")
        plt.show()