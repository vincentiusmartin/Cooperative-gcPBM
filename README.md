# Cooperative-gcPBM
The pipeline for analyzing cooperative genomic context protein binding microarray (cooperative gcPBM) data

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

Output:
1. Clean probe file with the fields mentioned above
2. Negative control probe file

Running the code:
- ETS1-ETS1: `python3 clean_file.py data/probe_files/raw/ETS1_ETS1.txt -k "ets1" -e "dist|weak" -g`
- ETS1-RUNX1:
  - ETS1 only chamber: `python3 clean_file.py data/probe_files/raw/ETS1_only.txt -k "all_clean_seqs" -n "negative_controls" -f`
  - ETS1-RUNX1 chamber: `python3 clean_file.py data/probe_files/raw/ETS1_RUNX1.txt -k "all_clean_seqs" -n "negative_controls" -f`
- RUNX1-ETS1:
  - RUNX1 only chamber: `python3 clean_file.py data/probe_files/raw/RUNX1_only.txt -k "all_clean_seqs" -n "negative_controls" -f`
  - RUNX1-ETS1 chamber: `python3 clean_file.py data/probe_files/raw/RUNX1_ETS1.txt -k "all_clean_seqs" -n "negative_controls" -f`


## ETS1-ETS1 analysis pipeline:
1. **Labeling the probe data for ETS1-ETS1:**
label_pr_ets_ets.py

2. **Generate training data ETS1-ETS1:**
traingen_ets_ets.py

3. **Generate Random Forest model ETS1-ETS1:**
genmodel_ets_ets.py

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
