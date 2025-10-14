import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess

global used_files
used_files = []

def train_and_get_errors(input_model, dataset, output_model, verification_file, duplicate_file, target):
    global used_files
    model_target_lut = {"VC4": ("32", "2", "16,12"), "PISP": ("16", "3", "32,32")}
    model_size, model_conv_layers, image_size = model_target_lut[target]
    args = ['python', 'train.py', dataset, output_model,
                             '--model-size', model_size,
                             '--model-conv-layers', model_conv_layers,
                             '--reduce-lr',
                             '--model-dropout', '0.1',
                             '--image-size', image_size,
                             '--batch-size', '8',
                             '--early-stopping',
                             '--duplicate-file', duplicate_file,
                             '--epochs', '150']
    if input_model is not None:
        args += ['--input-model', input_model]
    print("    Training...")
    result = subprocess.run(args, capture_output=True)

    args = ['python', 'verify_dataset.py', '--dataset', dataset, '--model', output_model, '--log', verification_file]
    print("    Verifying...")
    result = subprocess.run(args, capture_output=True)

    with open(verification_file, "r") as f:
        lines = f.readlines()
    worst = int(lines[-2].split()[-1])
    average = float(lines[-1].split()[-1])

    # Don't target a file that we've already chosen in the last "N" runs.
    N = 0
    for i in range(len(lines) - 2):
        file = lines[i].split()[0]
        if file not in used_files:
            break
    used_files.append(file)
    if len(used_files) > N:
        used_files = used_files[1:]

    return worst, average, file


def auto_train(input_model, dataset, duplicate_file, output_dir, weight, iterations, early_stopping, target):
    verification_name = "verification"
    verification_file = f"{verification_name}.txt"
    output_name = "model"
    output_model = f"{output_name}.keras"
    best_worst = 999999
    stopping_number = 0
    duplicate_name = os.path.splitext(duplicate_file)[0]

    for i in range(iterations):
        print("-" * 40)
        print(f"Training run {i} at {datetime.datetime.now()}:")
        worst, average, file = train_and_get_errors(input_model, dataset, output_model, verification_file, duplicate_file, target)

        if worst < best_worst:
            stopping_number = 0
            best_worst = worst
        else:
            stopping_number += 1
            if stopping_number >= early_stopping:
                print("Failure to progress - stopping")
                return best_worst

        input_model = output_model  # after first iteration, re-read the most recent output
        average = round(average, 4)
        print("    Saving results for: worst", worst, "average", average)
        average = str(average).replace('.', 'p')
        shutil.copy(output_model, f"{output_dir}/{output_name}_{worst}_{average}_{i}.keras")
        shutil.copy(duplicate_file, f"{output_dir}/{duplicate_name}_{worst}_{average}_{i}.txt")
        shutil.copy(verification_file, f"{output_dir}/{verification_name}_{worst}_{average}_{i}.txt")
        file = file.split('/')[-1]
        file = file.split(',')[:3]
        file = ','.join(file)
        with open(duplicate_file, "a") as f:
            print(file, weight, file=f)
        print(f"    Added weight {weight} to file {file}")

    return best_worst

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Where to load the dataset from")
    parser.add_argument("output_dir", type=Path, help="Folder to save results")
    parser.add_argument("--input-model", type=Path, help="Load a model to continue training", default=None)
    parser.add_argument("--duplicate-file", type=Path, help="File listing images to be included more than once", default="duplicates.txt")
    parser.add_argument("--iterations", type=int, help="Number of training iterations", default=50)
    parser.add_argument("-w", "--weight", type=int, help="Number of extra times to include difficult images", default=4)
    parser.add_argument("--early-stopping", type=int, help="Stop after this many iterations with no improvement", default=10)
    parser.add_argument("--clear-duplicates", action="store_true", help="Clear duplicates file", default=False)
    parser.add_argument("-t", "--target", type=str, help="Target platform, either PISP or VC4", required=True)
    args = parser.parse_args()

    if not args.target:
        raise ValueError("Please specify a target - either VC4 or PISP")
    target = args.target.upper()
    if target not in ("VC4", "PISP"):
        raise ValueError(f"Target {args.target} not recognised - use one of VC4 or PISP")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Target platform:", target)
    print("Dataset folder:", args.dataset)
    print("Starting with input model:", args.input_model or "None (training from scratch)")
    print("Using duplicate file:", args.duplicate_file)
    if os.path.isfile(args.duplicate_file):
        if args.clear_duplicates:
            with open(args.duplicate_file, "w"):
                pass
            print("    (File has been cleared)")
        else:
            with open(args.duplicate_file, "r") as f:
                lines = sum(1 for line in f if not line.strip().startswith('#'))
            print("    (File contains", lines, "lines)")
    else:
        with open(args.duplicate_file, "w"):
            pass
        print("    (Empty file created)")

    print("Outputs being copied to:", args.output_dir)
    print("Weight:", args.weight)
    print("Iterations:", args.iterations)

    worst = auto_train(args.input_model, args.dataset, args.duplicate_file, args.output_dir,
                       args.weight, args.iterations, args.early_stopping, target)
    print("Final worst-case error was:", worst)
