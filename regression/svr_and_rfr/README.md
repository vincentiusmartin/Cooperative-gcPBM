## Random Forest Regression and Support Vector Regression
Predicting transcription factor cooperativity from DNA sequences.

### To Run
With SLURM:
```
sbatch train_and_cv.sh <output directory path> <data config file path>
```
- output directory path: path to output results
- data config file path: path to a `.json` file containing paths to data (see `data-config.json`
for example).

This allows for a variety of grid searches with different feature sets to be executed 
simultaneously.

Without SLURM:
```
python automate_grid_search.py <job_id> <output_path> <ets1_ets1|ets1_runx1>
<random_forest_regression|support_vector_regression> <feature_1>,<feature_2>,...,<feature_n>
```
Descriptions for the positional arguments in the above command:
- `job_id` (str): arbitrary ID #, primarily for use in a SLURM job
- `output_path` (str): path to output results
- `experiment` (str): Either "ets1_ets1" or "ets1_runx1"
- `model` (str): either "random_forest_regression" or "support_vector_regression"
- `feature_list: comma separated list of features to include. See "feature_sets" in
`train_and_cv.sh` for examples. Eg. `distance,affinity,orientation`.

`automate_grid_search.py` runs a grid search with one feature set. The values of the grid search
are defined within the script.
