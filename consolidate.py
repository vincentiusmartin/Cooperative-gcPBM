"""Run this program after the SLURM job has finished to consolidate information from the main
task_{id}.json files into one file ("consolidated_info.json") where each task gets one dictionary
object with aggregate information.
"""
from copy import deepcopy
import glob
import json
import statistics
import sys


def main():
    if len(sys.argv) == 1:
        print("pass a path as an argument consolidate all task_{id}.json file data into one file.")
        return

    dir_path = sys.argv[1]
    files = glob.glob(f"{dir_path}task_*.json")

    with open(f"{dir_path}consolidated_info.json", "w") as f:
        f.write(json.dumps([]))

    if len(files) < 1:
        return
    for file_path in files:
        with open(file_path, "r") as f:
            json_list = json.load(f)

        if len(json_list) == 0:
            continue

        new_entry = deepcopy(json_list[0])
        del new_entry["cv_test_r2_mean"]
        del new_entry["cross_validation_test_r2"]
        del new_entry["random_state"]
        new_entry["file_path"] = file_path

        new_entry["cv_r2_means"] = []
        for entry in json_list:
            new_entry["cv_r2_means"].append(entry["cv_test_r2_mean"])

        new_entry["cv_r2_mean"] = 0
        new_entry["cv_r2_mean_std"] = 0

        if len(new_entry["cv_r2_means"]) > 1:
            new_entry["cv_r2_mean"] = statistics.mean(new_entry["cv_r2_means"])
            new_entry["cv_r2_mean_std"] = statistics.stdev(new_entry["cv_r2_means"],
                                                           new_entry["cv_r2_mean"])

        with open(f"{dir_path}consolidated_info.json", "r+") as f:
            json_list = json.load(f)
            json_list.append(new_entry)
            f.seek(0)
            f.write(json.dumps(json_list))
            f.truncate()


if __name__ == "__main__":
    main()
