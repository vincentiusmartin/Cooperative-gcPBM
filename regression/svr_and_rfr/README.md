## Random Forest Regression and Support Vector Regression
Predicting transcription factor cooperativity from DNA sequences.

### To Run
On SLURM:
`sbatch train_and_cv.sh <output directory path> <data config file path>`
- output directory path: path to output results
- data config file path: path to a `.json` file containing paths to data (see `data-config.json`
for example).
This allows for a variety of grid searches with different feature sets to be executed 
simultaneously.

An experiment may be run without SLURM using the following command:
` python automate_grid_search.py <job_id> <outputpath> <ets1_ets1|ets1_runx1>
<random_forest_regression|support_vector_regression> <feature_1>,<feature_2>,...,<feature_n>`
- job_id: arbitrary ID #, primarily for use in a SLURM job
- output path: path to output results
- experiment dataset: Either "ets1_ets1" or "ets1_runx1"
- model: either "random_forest_regression" or "support_vector_regression"
- feature list: comma separated list of features to include. See "feature_sets" in
`train_and_cv.sh` for examples. Eg. `distance,affinity,orientation`.

`automate_grid_search.py` runs a grid search with one feature set. The values of the grid search
are defined within the script.
