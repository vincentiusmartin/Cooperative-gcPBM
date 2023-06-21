# Cooperative-gcPBM
The pipeline for analyzing cooperative genomic context protein binding microarray (cooperative gcPBM) data.

All input and processed files, including figures, are available in: https://duke.box.com/s/cnbo6gjg223mtdun3cnemycwep414wgd

## Generate clean probe files
Code: `clean_file.py`

Take as input the raw probe files and generate csv files containing the required informations for the pipeline:
1. Name: Probe name
2. Sequence: Probe sequence
3. intensity: TF binding levels
4. type: Mutation type: wt (wild type), m1/m2 (one site mutated), m3 (two sites mutated). Or negctrl for negative controls.
5. rep: Replicates
6. ori: Orientation of the sequence

Output files:
1. A clean probe file with the fields mentioned above
2. A negative control probe file

Run:
- ETS1-ETS1: `python3 clean_file.py data/probe_files/raw/ETS1_ETS1.txt -k "ets1" -e "dist|weak" -g`
- ETS1-RUNX1:
  - ETS1 only chamber: `python3 clean_file.py data/probe_files/raw/ETS1_only.txt -k "all_clean_seqs" -n "negative_controls" -f`
  - ETS1-RUNX1 chamber: `python3 clean_file.py data/probe_files/raw/ETS1_RUNX1.txt -k "all_clean_seqs" -n "negative_controls" -f`
- RUNX1-ETS1:
  - RUNX1 only chamber: `python3 clean_file.py data/probe_files/raw/RUNX1_only.txt -k "all_clean_seqs" -n "negative_controls" -f`
  - RUNX1-ETS1 chamber: `python3 clean_file.py data/probe_files/raw/RUNX1_ETS1.txt -k "all_clean_seqs" -n "negative_controls" -f`


## ETS1-ETS1 analysis pipeline:
### 1. Label the probe data for ETS1-ETS1 ###

Description: Label each sequence as cooperative/ambiguous/independent

Code: `label_pr_ets_ets.py`

Run: `python3 label_pr_ets_ets.py data/probe_files/clean/ETS1_ETS1_pr_clean.csv -n data/probe_files/clean/ETS1_ETS1_neg_clean.csv -f`

Additional arguments: `python3 label_pr_ets_ets.py -h`

Output files:
1. `ets_ets_seqlabeled.csv`: Sequences with labels, this file is used as the main input for the subsequent analysis
2. `ETS1_ETS1_indiv.csv`: intensity for each combination of ETS1 binding to individual sites (i.e. m1+m2)
3. `ETS1_ETS1_two.csv`: intensity for each combination of ETS1 binding to two sites (i.e. wt)
4. `labeled_ets_ets_scatter.pdf`: scatter plot for cooperative vs. independent sequences

Example outputs, see: `data/analysis_files/ETS1-ETS1/labeled`

### 2. Generate training data ETS1-ETS1 ###

Description: Generate training data with all the features and labels for the sequences containing two ETS1 sites

Code: `traingen_ets_ets.py`

Run: `python3 traingen_ets_ets.py data/analysis_files/ETS1_ETS1/labeled/ets_ets_seqlabeled.csv -p data/sitemodels/ETS1.txt -k data/sitemodels/ETS1_kmer_alignment.txt`

Output files:
1. `train_ETS1_ETS1.tsv`: Training data for ETS1-ETS1
2. Three figure files with the distributions for distance, orientation, and strength features

Example outputs, see: `data/analysis_files/ETS1-ETS1/training`

### 3. Generate Random Forest model for ETS1-ETS1 ###
Code: `genmodel_ets_ets.py`

Run: `python3 genmodel_ets_ets.py data/analysis_files/ETS1_ETS1/training/train_ETS1_ETS1.tsv`

Note: `rf_param_grid` is currently hardcoded, please change the parameters directly in the code as needed

Output files:
1. `ETS1_ETS1_rfmodel.sav`: pickle file with the random forest model trained on ETS1-ETS1 data using distance, orientation, and strength features
2. `auc_all.png`: AUC curve with the model performances
3. `auc_all.log`: A text file with the mean accuracy, mean AUC, and confusion matrices for all the models tested.

Example outputs, see: `data/analysis_files/ETS1-ETS1/model`

## ETS1-RUNX1 and RUNX1-ETS1
### 1. Labeling the probe data for ETS1-RUNX1 ###

Description: Label each sequence as cooperative/ambiguous/independent

Code: `label_pr_ets_runx.py`

Run: `python3 label_pr_ets_runx.py`

Note: there are a lot of parameters for the script and currently they are still hardcoded, please check the header in `main`. To change between ETS1-RUNX1 and RUNX1-ETS1 please use the relevant commented part provided in the code.

Output files using ETS1 as the main TF and RUNX1 as the cooperator TF (i.e. ETS1-RUNX1):
1. `ets1_runx1_seqlbled.tsv`: Sequences with labels, this file is used as the main input for the subsequent analysis.
2. `ets1_runx1_main.csv`: Intensity for the chamber with the main TF alone.
3. `ets1_runx1_main_cooperator.csv`: Intensity for the chamber with the main TF in the presence of the cooperator TF.
4. `normalized_ets1_runx1.pdf`: scatter plot for cooperative vs. independent sequences using the normalized data.
5. `both_ori_plt_ets1_runx1.csv`: Each column represents the value used to plot (4).
6. `seq_er_intensity.csv`: Median binding intensity for the main TF alone; main + cooperator TFs both normalized and unnormalzied.

### 2. Generate training data ETS1-RUNX1 ###

Description: Generate training data with all the features and labels for the sequences containing ETS1 and RUNX1 sites

Code: `traingen_ets_runx.py`

Run: `python3 traingen_ets_runx.py`

Output files (for ETS1-RUNX1):
1. `train_ets1_runx1.tsv`: Training data for ETS1-RUNX1
2. Three figure files with the distributions for distance, orientation, and strength features

Example outputs, see: `data/analysis_files/ETS1-RUNX1/training`

### 3. Generate Random Forest model ETS1-RUNX1 ###

Code: `genmodel_ets_runx.py`

Run:
- ETS1-RUNX1: `python3 genmodel_ets_runx.py data/analysis_files/ETS1_RUNX1/training/train_ets1_runx1.tsv`
- RUNX1-ETS1: `python3 genmodel_ets_runx.py data/analysis_files/RUNX1_ETS1/training/train_runx1_ets1.tsv`

Output files:
1. `ETS1_RUNX1_rfmodel.sav`: pickle file with the random forest model trained on ETS1-RUNX1 data using distance, orientation, and strength features
2. `auc.png`: AUC curve with the model performances
3. `auc.log`: A text file with the mean accuracy, mean AUC, and confusion matrices for all the models tested.

Example outputs, see: `data/analysis_files/ETS1-RUNX1/model`

## Shape analysis for ETS1-ETS1 or ETS1-RUNX1

### 1. Generate Random Forest model using sequence and shape features ###

The code requires DNAShape R package and imported using `rpy2`. Please install the package as described in: https://bioconductor.org/packages/release/bioc/html/DNAshapeR.html

Code: `gen_posmdl.py`

Run:
- ETS1-ETS1: `python3 gen_posmdl.py data/analysis_files/ETS1_ETS1/training/train_ETS1_ETS1.tsv -a site_str -b site_wk -s relative -r -o`
- ETS1-RUNX1: `python3 gen_posmdl.py data/analysis_files/ETS1_RUNX1/training/train_ets1_runx1.tsv -a ets1 -b runx1 -s positional`
- RUNX1-ETS1:`python3 gen_posmdl.py data/analysis_files/RUNX1_ETS1/training/train_runx1_ets1.tsv -a runx1 -b ets1 -s positional`

Output files:
1. `rfposmodel.sav`: A pickle file with the random forest model trained on ETS1-ETS1 data using distance, orientation, shape, and sequence features
2. `auc_posfeatures.pdf`: A figure with the ROC curve showing the model performances
3. `auc_all.log`: A text file with the mean accuracy, mean AUC, and confusion matrices for all the models tested.

### 2. Shape analysis for ETS1-ETS1 ###

Create summary motif and shape figures for all sequences in the training data, also outputs the list of sequences for each configuration.

Code: `shape_analysis.py`

Run:
- ETS1-ETS1: `python3 shape_analysis.py data/analysis_files/ETS1_ETS1/training/train_ETS1_ETS1.tsv -p site_str_pos,site_wk_pos`
- ETS1-RUNX1: `python3 shape_analysis.py data/analysis_files/ETS1_RUNX1/training/train_ets1_runx1.tsv -p ets1_pos,runx1_pos`
- RUNX1-ETS1: `python3 shape_analysis.py data/analysis_files/RUNX1_ETS1/training/train_runx1_ets1.tsv -p runx1_pos,ets1_pos`

Example outputs, see:
- ETS1-ETS1: `data/analysis_files/ETS1_ETS1/shape_out`
- ETS1-RUNX1: `data/analysis_files/ETS1_RUNX1/shape_out`
