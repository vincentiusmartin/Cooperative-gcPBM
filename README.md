# Cooperative-gcPBM
The pipeline for analyzing cooperative genomic context protein binding microarray (cooperative gcPBM) data

## Generate clean probe files
Code: `clean_file.py`

Take as input the raw probe files and generate csv files containing the required informations for the pipeline:
Probe Name,
Sequence,
Probe intensity, type (wt/m1/m2/m3), replicate ids, orientation (o1 for original)

Output:
1. Clean probe file with the fields mentioned above
2. Negative control probe file

Running the code:
- ETS1-ETS1: `python3 clean_file.py data/probefiles/raw/ETS1_ETS1.txt -k "ets1" -e "dist|weak" -g`
- ETS1-RUNX1:
  - ETS1 only chamber: `python3 clean_file.py data/probefiles/raw/ETS1_only.txt -k "all_clean_seqs" -n "negative_controls" -f`
  - ETS1-RUNX1 chamber: `python3 clean_file.py data/probefiles/raw/ETS1_RUNX1.txt -k "all_clean_seqs" -n "negative_controls" -f`
- RUNX1-ETS1:
  - RUNX1 only chamber: `python3 clean_file.py data/probefiles/raw/RUNX1_only.txt -k "all_clean_seqs" -n "negative_controls" -f`
  - RUNX1-ETS1 chamber: `python3 clean_file.py data/probefiles/raw/RUNX1_ETS1.txt -k "all_clean_seqs" -n "negative_controls" -f`


## ETS1-ETS1:
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
