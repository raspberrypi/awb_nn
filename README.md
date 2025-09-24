## Instructions

### Setup the environment

py3exiv2 needs:

Install py3evix2 dependencies:
```bash
sudo apt install libexiv2-dev libboost-python-dev
```

Create a python virtual environment and install dependencies:

```bash
python -m venv .venv --system-site-packages
.venv/bin/pip install tensorflow opencv-python py3exiv2 rawpy
```

### Gathering data

To gather the raw images use `snapper.py` to gather images to train the model.
It starts in fullscreen which works well with a display attached to the pi.

```bash
python snapper.py --user <username> --output <output-directory>
```

`snapper.py` needs to be run outside of the virtual envirnment.
Run `python snapper.py --help` to see all the options.
The `<username>` is only used to name the images so it could be used to sort the images into categories.
For example, where they were taken.

`snapper.py` will output images to the output directory as both JPEG and DNG files. The filenames will be formatted as `<username>,<sensor>,<id>.dng` and `<username>,<sensor>,<id>.jpg`. The id counts up from 0 by default so use `--initial-scene-id <id>` to prevent overwriting the old images.

### Labelling data

The rest of the python scripts need to be run from the virtual environment so activate it with
```bash
source .venv/bin/activate
```

The data needs to be labeled with the correct colour gains using `annotator.py` There are 3 ways to label each image:
- Draw a rectangle on a patch of gray in the image and the gains will be calculated automatically.
- Select a colour temperature with the slider. Use the Purple/Green slider if the correct white balance does not lie on the colour temperature curve.
- Type in the red and blue gains manually

Activate the virtual environment and run the annotoator:

```bash
python annotator.py -i <directory>
```

`annotator.py` will rename only the DNG file to `<username>,<sensor>,<id>,<red_gain>,<blue_gain>.dng`

### Preparing the dataset

The images then need to be converted to PNG files so that they can be loaded by TensorFlow.
Use `converter.py` to convert the DNG files to PNG files. It outputs files in the format `<username>,<sensor>,<id>,<true_temperature>,<lux>,<camera_temperature>.dng`.
`<true_temperature>` is the colour temperature it was labeled as and `<camera_temperature>` is what the camera predicted when it took the image. This is useful for comparing the results of the model.

To check that the images have been colour balanced correctly, run:
```bash
python converter.py <input-dir> <output-dir> --resize 640,480 --colour-test
```
That will output the images colour balanced correctly to check them.
These images are only for checking the colour balance and are not used later.

Once the images have the correct colour balance, run
```bash
python converter.py <input-dir> <output-dir> --split <split> --resize <width>,<height>
```

- `input-dir` is where the DNG files from annotator are located
- `--split` is how many of the images are moved to the test dataset. 0.2 is a good choice and it must be specified to train the model.
- `--resize` is optional but reduces file size and trains the model faster. It should be at least 32,32

Run `python converter.py --help` to see all the options.

### Training the model

The Pi 5 has more AWB zones than other Pis so its model needs to be trained separately.
It takes a few minutes to train the model on a Pi 5.
The training will stop early if it stops improving.
Run `python train.py --help` to see what all the options do.

Pi 5:
```bash
python train.py <dataset> model_pisp.keras --model-size 16 --early-stopping --reduce-lr --model-dropout 0.1 --image-size 32,32
```

Pi 4 and older:
```bash
python train.py <dataset> model_vc4.keras --model-size 32 --model-conv-layers 2 --early-stopping --reduce-lr --model-dropout 0.1 --image-size 16,12
```

### Testing the model

The output of the model can be tested on the DNG files:
```bash
python verify.py -i <image> --model <model>
```

The output of the model is displayed along with what the image was labelled as and the original estimate from the camera.

### Convert the model

The model needs to be converted to TFLite format before it can run on a Raspberry Pi.
It can be quantized to float16 or int8 to reduce the file size without a noticeable loss in accuracy.

To produce the smallest model run
```bash
python model_converter.py <model> --quantize int8
```

This will output a TFLite model which can be run on a Raspberry Pi.

### Running the model on a Pi

The neural network AWB is disabled by default.
To use it, run
```bash
python gen_tuning.py --sensor <sensor> --isp <isp> --model <model>
```
Where `sensor` is the name of the camera's sensor (e.g. imx708) and `isp` is `pisp` on a Pi 5 and `vc4` on any other Pi. Run `python gen_tuning.py --help` to see all the options.

This will generate a new tuning file which can be used by rpicam-apps with the `--tuning-file` argument and in picamera2 with `Picamera2(tuning="<filename>.json")`
