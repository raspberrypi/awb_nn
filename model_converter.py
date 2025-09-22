import tensorflow as tf
import argparse
from pathlib import Path
from train import MiredLoss, ClassesToTemp, PreprocessImage

def convert_model(model_path: Path, output_path: Path | None = None, quantization: str | None = None):
    model = tf.keras.models.load_model(model_path, custom_objects={
        "MiredLoss": MiredLoss,
        "ClassesToTemp": ClassesToTemp,
        "PreprocessImage": PreprocessImage
    })
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization:
        if quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        else:
            raise ValueError(f"Invalid quantization: {quantization}")

    tflite_model = converter.convert()
    if output_path is None:
        output_path = model_path.with_suffix(".tflite")

    with open(output_path, "wb") as f:
        f.write(tflite_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path, help="The path to the model to convert", metavar="model-path")
    parser.add_argument("-o", "--output", type=Path, help="The path to save the converted model", default=None)
    parser.add_argument("--quantize", type=str, help="Quantize the model", default=None, choices=["float16", "int8"])
    args = parser.parse_args()

    convert_model(args.model_path, args.output, args.quantize)
