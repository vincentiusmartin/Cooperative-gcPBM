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

    for key, value in config.items():
        if key == "kernel_widths":
            config["kernel_widths"] = [[str(w) for w in k] for k in value]
            continue

        config[key] = [str(w) for w in value]

    for i in config["num_layers"]:
        kernel_widths = config["kernel_widths"][:int(i)]
        temp = deepcopy(config)
        temp["num_layers"] = [i]
        temp["kernel_widths"] = (",".join(x) for x in itertools.product(*kernel_widths))

        arg_set += (f"{' '.join(k)};" for k in itertools.product(*temp.values()))

    print(" ".join(arg_set))
