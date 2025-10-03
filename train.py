import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from data import TEMP_CLASSES
from dataset import create_dataset_pair, temp_classes_tensor
from pathlib import Path
import argparse

@tf.keras.utils.register_keras_serializable(name="TempLoss")
class MiredLoss(tf.keras.losses.Loss):
    """
    Loss function for temperature prediction.

    Calculates the mean absolute error between the predicted and true temperature in mireds / 1000.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(1000000 / y_pred - 1000000 / y_true)) / 1000

@tf.keras.utils.register_keras_serializable(name="ClassesToTemp")
class ClassesToTemp(tf.keras.layers.Layer):
    """
    Convert classes to a temperature.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, classes: tf.Tensor) -> tf.Tensor:
        """
        Convert classes to a temperature.
        """
        return classes_to_temp(classes)


class PreprocessImage(tf.keras.layers.Layer):
    """
    Layer to preprocess an image.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, image, lux):
        """
        Expects a tuple (image, lux).
        """
        green = image[..., 1]
        green = tf.clip_by_value(green, 0.1, 1)
        red_ratio = tf.clip_by_value(image[..., 0] / green, 0.01, 1)
        blue_ratio = tf.clip_by_value(image[..., 2] / green, 0.01, 1)

        red_channel = (tf.math.log(red_ratio) + 1.0) / 2.0
        blue_channel = (tf.math.log(blue_ratio) + 1.0) / 2.0
        lux_norm = tf.math.log(lux) / 10.0
        lux_norm = tf.reshape(lux_norm, (-1, 1, 1))
        green_channel = tf.ones_like(green) * lux_norm

        new_image = tf.stack([red_channel, green_channel, blue_channel], axis=-1)
        new_image = tf.clip_by_value(new_image, 0, 1)

        return new_image



def classes_to_temp(classes: tf.Tensor) -> tf.Tensor:
    """
    Convert classes to a temperature.
    """
    weighted = classes * classes * temp_classes_tensor
    total_weight = tf.reduce_sum(classes * classes, axis=1)
    return tf.reduce_sum(weighted, axis=1) / total_weight


def create_model(size: int, dropout: float | None = None, input_shape: tuple[int, int] = (32, 32), conv_layers: int = 3) -> keras.Model:
    inp = keras.layers.Input(shape=input_shape + (3,))
    lux = keras.layers.Input(shape=())
    x = PreprocessImage()(inp, lux)
    x = keras.layers.RandomFlip("horizontal")(x)
    x = keras.layers.RandomRotation(0.1)(x)
    x = keras.layers.RandomZoom(0.1)(x)

    for i in range(conv_layers):
        x = keras.layers.Conv2D(size * 2 ** i, (3, 3), activation="relu")(x)
        if i < conv_layers - 1:
            x = keras.layers.MaxPooling2D((2, 2))(x)
        else:
            x = keras.layers.Flatten()(x)

    if dropout is not None:
        x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dense(TEMP_CLASSES, activation="softmax")(x)

    x = ClassesToTemp()(x)

    model = keras.Model(inputs=[inp, lux], outputs=x)
    model.compile(optimizer="adam", loss=MiredLoss())
    return model


def train_model(model: tf.keras.Model,
        trains_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        log_dir: Path | None = None,
        epochs: int = 25,
        early_stopping: bool = True,
        reduce_lr: bool = True,
        input_shape: tuple[int, int] = (32, 32)
    ) -> None:
    """
    Train a model.

    Args:
        model (Model): The model to train.
        trains_ds (Dataset): The training dataset.
        val_ds (Dataset): The validation dataset.
        tensorboard_callback (TensorBoard): The tensorboard callback.
        epochs (int): The number of epochs to train for.
    """
    callbacks = []
    checkpoint_filepath = "best_model.weights.h5"
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
        ))
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            mode="min",
            patience=25,
            restore_best_weights=True,
            start_from_epoch=10
        ))
    if reduce_lr:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
        ))
    if log_dir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))

    model.fit(trains_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    model.load_weights(checkpoint_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="Where to load the dataset from", metavar="dataset-dir")
    parser.add_argument("output_model", type=Path, help="Where to save the trained model", metavar="output-model")
    parser.add_argument("--epochs", type=int, help="The number of epochs to train for", default=50)
    parser.add_argument("--batch-size", type=int, help="The batch size to use for training", default=32)
    parser.add_argument("--log-dir", type=Path, help="Where to save the logs", default=Path("logs"))
    parser.add_argument("--input-model", type=Path, help="Load a model to continue training", default=None)
    parser.add_argument("--model-size", type=int, help="The size of the model", default=32)
    parser.add_argument("--model-conv-layers", type=int, help="The number of convolutional layers to use", default=3)
    parser.add_argument("--model-dropout", type=float, help="The dropout rate to use", default=None)
    parser.add_argument("--early-stopping", action="store_true", help="Stop training early if the validation loss does not improve", default=False)
    parser.add_argument("--reduce-lr", action="store_true", help="Reduce learning rate during training if the loss stops improving", default=False)
    parser.add_argument("--image-size", type=str, help="The shape of the input images", default="32,32")
    parser.add_argument("--lr", type=float, help="Initial learning rate", default=1e-3)
    parser.add_argument("--duplicate-file", type=Path, help="File listing images to duplicate in the training set")
    args = parser.parse_args()

    try:
        image_size = tuple(int(x) for x in args.image_size.split(","))
        if len(image_size) != 2:
            raise ValueError()
        image_shape = tuple(reversed(image_size))
    except ValueError:
        raise argparse.ArgumentError("Input shape must be a pair of integers in the format WIDTH,HEIGHT")

    log_dir = args.log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    trains_ds, val_ds = create_dataset_pair(args.dataset_dir, image_shape, batch_size=args.batch_size,
                                            duplicate_file=args.duplicate_file)

    if args.input_model is not None:
        model = tf.keras.models.load_model(args.input_model, custom_objects={
            "MiredLoss": MiredLoss,
            "ClassesToTemp": ClassesToTemp,
            "PreprocessImage": PreprocessImage
        })
    else:
        model = create_model(args.model_size, args.model_dropout, input_shape=image_shape, conv_layers=args.model_conv_layers)

    model.optimizer.learning_rate.assign(args.lr)


    model.summary()
    train_model(
        model,
        trains_ds,
        val_ds,
        log_dir,
        epochs=args.epochs,
        early_stopping=args.early_stopping,
        reduce_lr=args.reduce_lr,
        input_shape=image_shape,
    )
    model.save(args.output_model)
