# Convolutional Neural Network for cooperativity prediction

### To Run
With SLURM:
```
sbatch -p compsci-gpu --gres=gpu:1 train_and_evaluate.sh <gridsearch config path> <data config path>
```
- gridsearch config path: path to the file that specifies the gridsearch we want to run. (All of the
`.json` files in the `experiment_config_files` directory are examples.)
- data config path: path to a file config file specifying where the training and labeled data is. 
`regression/deep_learning/data-config.json` is an example.

Without SLURM:
```
python regression/deep_learning/scripts/experiment.py
<job_id> <output_path> <data_config> <experiment_name> ... <test: TRUE|FALSE>
```
The required arguments are laid out below below:
- `job_id`: arbitrary ID #, primarily for use in a SLURM job
- `output_path`: path to put output of experiment
- `data_config`: path to config file which specifies location of data
- `experiment_name`: name of experiment
- `num_layers` (int): number of convolutional layers
- `mers` (int): one-hot encode sequence as 1-mers, 2-mers, or 3-mers. Value is specified as `1`, `2`, or `3`.
- `batch_size` (int): batch size used during training the neural network
- `kernel_widths` (str; comma separated integer values with no spaces): for each convolutional layer, specify width of the kernel
  (eg. for a 4 layer CNN (`3,4,5,6`, would indicate first through fourth convolutional layers 
will have kernel widths 3, 4, 5, and 6 respectively.)
- `include_affinities`: inject binding strengths as input to fully-connected layer
- `pool` (boolean): use pooling or not after convolutional layers
- `dropout_rate` (float): fraction to specify dropout rate
- `weight_decay` (float): L2 regularization for the neural network
- `lr` (float): learning rate
- `conv_filters` (int): number of convolutional filters at each layer (set for all layers)
- `fc_node_count` (int): number of nodes at the first fully-connected layer
- `test` (boolean): whether to use the test data splits (`TRUE`) or validation data splits (`FALSE`)