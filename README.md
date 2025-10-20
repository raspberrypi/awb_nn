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
.venv/bin/pip install tensorflow opencv-python py3exiv2 rawpy matplotlib
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

To convert the images into a dataset that can be used for training, use
```bash
python converter.py <input-dir> <output-dir> --split <split> --target <target>
```
where

- `input-dir` is where the DNG files from annotator are located.
- `output-dir` is the folder for the new dataset that can be used for training.
- `--split` is proportion of images that are moved to the test dataset. We have found that if you don't have that many images (and in this context 1000 counts as "not that many") a value of zero often works best. The reason is that you otherwise risk having images in the test set that can't be improved because there are no others that are similar enough in the training set. Obviously you will need to proceed carefully here and test the final networks thoroughly.
- `--target` should be either `vc4` (Pi 4 and older models) or `pisp` (Pi 5 and newer models). You will need to generate a separate dataset for each of the two platforms. The datasets are very much smaller than the image folders they are created from.

Run `python converter.py --help` to see all the options.

### Training the model

The best way to train the model is to use the `auto_train.py` script. You will need to run it once for the VC4 (Pi 4 and older) platform, and once for the PiSP (Pi 5) platform. This script runs the training procedure a number of times, on each occasion biasing the training towards images that have been showing the worst case errors. It could take typically about an hour to run on a Pi 5, and it can make sense to repeat the training multiple times if you have time.

To train a model, use
```bash
python auto_train.py <dataset> <output-dir> --target <target> [--duplicate-file <duplicate-file>] [--clear-duplicates]
```
where

- `<dataset>` is the dataset to use for training. Be sure to use a dataset created for the chosen target platform.
- `<output-dir>` is a folder where the results of each incremental training are written.
- `--target` lists the target platform, either `vc4` or `pisp`, and which should match your dataset.
- `--duplicate-file` names the file listing how many times particular images are re-used in training. The file format is the filename, followed by the count value, one file per line. It should be sufficient just to list the start of the image name, consisting of the `<user>,<sensor>,<id>` part. The default filename is `duplicates.txt`.
- `--clear-duplicates` instructs the script to clear the duplicates file when it starts, otherwise you can use your own duplicates file when you start the training process.
- `--input-model` names a model as a starting point for incremental training. Otherwise, training starts from scratch.

`auto_train.py` runs the training multiple times. Each run generates three output files - a "verification" file (listing images with their associated error, sorted from worst to best), a "duplicates" file (listing the images repeated more than once in the training) and a "model" file (the Keras model at the end of the training run). These files are all named with the worst error, average error, and run sequence number in the filename, separated by underscores. For example, the file
```
model_12_2p6003_31.keras
```
has a worst case error of 12, an average error of 2.6003 and was produced after training run 31.

The "verification" file in particular is useful for checking whether particular images were performing poorly or not.

Example:
```bash
python auto_train.py pisp-dataset pisp-output-0 --clear-duplicates --target pisp
```
will use the folder `pisp-dataset` for training, and write all the results to `pisp-output-0`. The duplicates file will be cleared when it starts, and it will train from scratch. It is training for the PiSP (Pi 5) platform.

#### Using your own Training Images

If you have your own images, you can annotate and convert them to a dataset as described above. You can train a network entirely on your own images, or you can add them to our datasets. As the performance on your own images may be particularly important, you may wish to create a duplicates file listing them (and avoid using the `--clear-duplicates` flag when you start training).

You can train from scratch, or incrementally from a previous model (using the `--input-model` parameter).

#### Training Tips

Training is stochastic procedure so it's often helpful to run `auto_train.py` multiple times - perhaps overnight - and then assess which of the models produced is the best.

You may find that `auto_train.py` can sometimes get stuck when the model has been trained into an area of the model's parameter space where it can no longer improve the worst case model. This problem can be mitigated by entering the problem image - on which it seems stuck - into the initial duplicates file (and obviously avoid the `--clear-duplicates` flag). As you repeat this procedure, you may find more images that need entering into your initial duplicates file, or you may need to increase the weight of images that are already listed.

The aim is to reach a point where you can run `auto_train.py` multiple times and, with reasonably reliability, it will generate good output models.

At the end of training, it's always important to test the models thoroughly on the target device, particularly as we currently have insufficient images to create a really large dataset with an effective training/validation split.

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
python gen_tuning.py --sensor <sensor> --target <target> --model <model>
```
Where `sensor` is the name of the camera's sensor (e.g. imx708) and `target` is `pisp` on a Pi 5 and `vc4` on any other Pi. Run `python gen_tuning.py --help` to see all the options.

This will generate a new tuning file which can be used by rpicam-apps with the `--tuning-file` argument and in picamera2 with `Picamera2(tuning="<filename>.json")`
