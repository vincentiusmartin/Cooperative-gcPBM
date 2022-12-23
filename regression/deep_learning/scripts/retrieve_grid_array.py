"""This program is used by the SLURM job to get a bash array in which each element is a set of
arguments for one run of 'train_and_evaluate.py`.
"""
from copy import deepcopy
import itertools
import json
import sys

if __name__ == "__main__":
    config_path = sys.argv[1]

    if not config_path:
        raise FileNotFoundError(f"The file, {config_path} does not exist.")

    with open(config_path, "r") as f:
        config = json.load(f)

    arg_set = []

    config["kernel_widths"] = [[str(w) for w in k] for k in config["kernel_widths"]]

    for key, array in config.items():
        if key == "kernel_widths":
            config["kernel_widths"] = [[str(w) for w in k] for k in array]
            continue

        config[key] = [str(w) for w in array]

    arg_count = 0
    arg_list = []
    for i in config["num_layers"]:
        kernel_widths = config["kernel_widths"][:int(i)]
        temp = deepcopy(config)
        temp["num_layers"] = [i]
        temp["kernel_widths"] = (",".join(x) for x in itertools.product(*kernel_widths))

        arg_list += itertools.product(*temp.values())
    arg_set = (f"{' '.join(k)};" for k in arg_list)

    if not len(sys.argv) > 2:
        print(" ".join(arg_set))
    else:
        print(f"sbatch -p compsci-gpu --array=0-{len(arg_list)-1}%32 --gres=gpu:1 train_and_evaluate.sh {config_path} data-config.json")
