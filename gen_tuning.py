import argparse
from pathlib import Path
import json
import os
from pretty_print import pretty_print

search_path = [
    Path("/usr/local/share"),
    Path("/usr/share"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", type=str, help="The sensor to generate the tuning for", required=True)
    parser.add_argument("--model", type=str, help="The model to use for the tuning", required=False)
    parser.add_argument("--output", type=str, help="The output file to save the tuning to", required=False)
    parser.add_argument("-t", "--target", type=str, help="The target platform. Use pisp for Pi 5, vc4 for other Pis", required=True, choices=["pisp", "PISP", "vc4", "VC4"])
    args = parser.parse_args()

    output = args.output if args.output else args.sensor + ".json"

    target = args.target.lower()
    for path in search_path:
        tuning_file = path / Path(f"libcamera/ipa/rpi/{target}") / f"{args.sensor}.json"
        if tuning_file.exists():
            break
    else:
        raise FileNotFoundError(f"Tuning file for {args.sensor} not found")

    with open(tuning_file, "r") as f:
        tuning = json.load(f)

    print(f"Tuning file loaded from {tuning_file}")

    found_bayes = False
    found_nn = False
    found_nn_enabled = False

    for algorithm in tuning["algorithms"]:
        if "rpi.awb" in algorithm:
            found_bayes = True
            awb = algorithm["rpi.awb"]
            del algorithm["rpi.awb"]
            algorithm["disable.rpi.awb"] = awb
        if "disable.rpi.nn.awb" in algorithm:
            found_nn = True
            awb = algorithm["disable.rpi.nn.awb"]
            del algorithm["disable.rpi.nn.awb"]
            algorithm["rpi.nn.awb"] = awb
            if args.model is not None:
                algorithm["rpi.nn.awb"]["model"] = os.path.abspath(args.model)
        if "rpi.nn.awb" in algorithm:
            found_nn_enabled = True

    if not found_nn:
        print("WARNING: disable.rpi.nn.awb not found in tuning file - neural network AWB may already be enabled")

    with open(output, "w") as f:
        if target == "vc4":
            custom_elems = {"table": 16, "luminance_lut": 16}
        elif target == "pisp":
            custom_elems = {"table": 32, "luminance_lut": 32}
        f.write(pretty_print(tuning, custom_elems=custom_elems))

    print(f"Tuning file saved to {output}")

