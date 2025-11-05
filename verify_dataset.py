import numpy as np
import math
import argparse
from pathlib import Path
import cv2
import glob
import tensorflow as tf
from train import MiredLoss, ClassesToTemp, PreprocessImage

def predict(image, model, lux):
    image_shape = model.input_shape[0][1:3]
    image = image.astype(np.float32) / 65535
    lux = float(lux)
    input_image = cv2.resize(image, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_AREA)
    input_image = np.expand_dims(input_image, axis=0)
    input_lux = np.expand_dims(lux, axis=0)
    temp = model.predict([input_image, input_lux])[0]
    return temp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, help="Folder of images to test", required=True)
    parser.add_argument("--model", type=Path, help="The model to test. Must be a keras model.", required=True)
    parser.add_argument("--log", type=Path, help="Log file to output the results")

    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, custom_objects={
        "MiredLoss": MiredLoss,
        "ClassesToTemp": ClassesToTemp,
        "PreprocessImage": PreprocessImage
    })
    model.summary()

    results = []
    for filename in args.dataset.rglob("*.png"):
        filename = str(filename)
        image = cv2.imread(filename, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image.max(), image.min())
        filename_parts = filename.split(',')
        real_temp, lux, _ = filename_parts[3:]

        model_temp = predict(image, model, lux)
        real_temp = int(real_temp)
        # Model doesn't produce values outside 2800K to 7600K, so don't penalise it further than
        # those values.
        real_temp = min(max(real_temp, 2800), 7600)
        model_temp = int(model_temp)
        error_K = int(abs(model_temp - real_temp))
        error_mireds = int(abs(1000000 / model_temp - 1000000 / real_temp))
        error_percent = int(100 * abs(model_temp - real_temp) / real_temp)
        results.append({"filename": filename, "real": real_temp,
                        "model": model_temp, "error_K": error_K, "error_mireds": error_mireds, "error_percent": error_percent})
        print(f"Results for {filename} : {real_temp} {model_temp} "
              f"error {error_K} {error_mireds}")

        temps = [real_temp, model_temp]

    results.sort(key=lambda d: d["error_mireds"], reverse=True)
    print("-" * 40)
    print("Sorted results:")
    output = ""
    avg = 0

    for r in results:
        if r["error_mireds"] > 2:
            output += f"{r['filename']} : {r['model']} should be {r['real']}, error {r['error_mireds']} ({r['error_percent']}%)\n"
        avg += r["error_mireds"]

    avg /= len(results)
    output += f"Worst error: {results[0]['error_mireds']}\n"
    output += f"Average error: {avg}\n"
    print(output, end="")
    if args.log:
        with open(args.log, "w") as f:
            f.write(output)
        print("Results written to", args.log)
