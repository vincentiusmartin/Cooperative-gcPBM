# Cooperative-gcPBM
The pipeline for analyzing cooperative genomic context protein binding microarray (cooperative gcPBM) data.

All input and processed files, including figures, are available in: https://www.dropbox.com/sh/x136g6plm2i6i7q/AAA-KAGtxAMu1hpV50dFi8ZEa?dl=0

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
2. `negdf.csv`: median intensity for each negative control sequence
3. `ets1_ets1_indiv.csv`: intensity for each combination of ETS1 binding to individual sites (i.e. m1+m2)
4. `ets1_ets1_two.csv`: intensity for each combination of ETS1 binding to two sites (i.e. wt)
5. `lbled_both.csv`: each row contain labels for each probe in orientations o1, o2, and also a label taking into account labels in both orientations
6. `ets_ets_m1m2m3wt.csv`: Median binding intensity for each sequence type
7. `lbled_o1_selected.csv`: Median intensity from combinations of binding to individual and two sites in orientation o1
8. `labeled_ets_ets_scatter.pdf`: scatter plot for cooperative vs. independent sequences

Example outputs, see: `data/analysis_files/ETS1-ETS1/labeled`

### 2. Generate training data ETS1-ETS1 ###

Description: Generate training data with all the features and labels for the sequences containing two ETS1 sites

Code: `traingen_ets_ets.py`

Run: `python3 traingen_ets_ets.py data/analysis_files/ETS1_ETS1/labeled/ets_ets_seqlabeled.csv -p data/sitemodels/ets1.txt -k data/sitemodels/Ets1_kmer_alignment.txt`

Output files:
1. `train_ets1_ets1.tsv`: Training data for ETS1-ETS1
2. Three figure files with the distributions for distance, orientation, and strength features

Example outputs, see: `data/analysis_files/ETS1-ETS1/training`

### 3. Generate Random Forest model for ETS1-ETS1 ###
Code: `genmodel_ets_ets.py`

Run: `python3 genmodel_ets_ets.py data/analysis_files/ETS1_ETS1/training/train_ets1_ets1.tsv`

Note: `rf_param_grid` is currently hardcoded, please change the parameters directly in the code as needed

Output files:
1. `ets1_ets1_rfmodel.sav`: pickle file with the random forest model trained on ETS1-ETS1 data
2. `auc_all.png`: AUC curve with the model performance

4. **Generate Random Forest model ETS1-ETS1 using sequence features:**
gen_posmdl.py

5. **Shape analysis for ETS1-ETS1:**
shape_analysis.py

## Ets1-Runx1
1. **Labeling the probe data for Ets1-Runx1:**
label_pr_ets_ets.py

2. **Generate training data Ets1-Runx1:**
traingen_ets_ets.py

3. **Generate Random Forest model Ets1-Runx1:**
genmodel_ets_ets.py

4. **Generate Random Forest model Ets1-Runx1 using sequence features:**
gen_posmdl.py

5. **Shape analysis for Ets1-Runx1:**
shape_analysis.py

## Make a scatter boxplot
scatter_boxplot.py
