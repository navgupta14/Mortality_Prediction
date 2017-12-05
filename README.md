# Mortality_Prediction

The code for Mortality Prediction using latent variable model has 3 parts.

1. Extracting data from MIMIC-III using SQL scripts: The data from MIMIC-III can be loaded into PostgreSQL by following the steps mentioned in the Mimic Code repository [1] and [2]. Once the data has been loaded into PostgreSQL, run the scripts present in the scripts/ directory of the project to generate the necessary csv files. 

2. Generate features from the extracted data: The extracted data is present in the form of csv files. The baseline features are present in the baseline_features.csv for each hadm_id(admission). The notes are present in raw text form in raw_notes.csv file. These raw notes have to be processed to transform them into features. These steps are carried out by running the following Python scripts in order:
    a. code/preprocessing/cleanup_notes.py - preprocesses the notes like removes non-alphanumerics, fix spaces, removes stop words etc.
    b. code/preprocessing/consolidate_notes.py - each admission id can have multiple notes, this consolidates all notes for a particular hadm_id.
    c. code/classifier/lda_module.py - runs lda on the notes to find the topic distributions.
    d. code/preprocessing/merge_baseline_text_features.py - this merges the structured features with the lda distributions to form final feature set.

3. After these steps, combined features (baseline and LDA), are written to combined_features.csv file.
Run classification algorithms: Using the features present in the combined_features.csv, run classification algorithms by running the script code/classifier/classification.py.

Steps 2 and 3 can be executed automatically by running the shell script getFeaturesAndRunClassifier.sh

References:
[1]https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/postgres
[2]https://github.com/rvpradeep/Mortality-Prediction

