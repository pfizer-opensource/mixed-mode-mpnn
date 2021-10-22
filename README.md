# MPNNs for Modeling Retention Times from Mixed Mode Chromatography / ESI Negative MS Detection
This repository contains code used to predict retention times given chemical structures (in the form of SMILES).

## Environment
The script `environment_setup.sh` contains instructions for setting up a conda environment with the appropriate versions of Tensorflow and DeepChem needed to run the models.

## Data
The raw data used to train and evaluate the model is tracked with Git-LFS in `data/combined_time.csv`. Each row describes the structure (SMILES in column `smiles`) and experimental results for a given compound. The column `neg RT (min)` in this file records the retention time for each compound.

The file `data/classes.py` contains a list of compound classes used to bucket the compounds in `data/combined_time.csv`.

## Jupyter Notebooks
The notebooks in the `notebooks` folder illustrate the process of training and evaluating the MPNN models as well as the steps to create the figures in our publication.
1. `DeepChem01.ipynb`: Train and test a message passing neural network on the dataset (random split)
2. `Manuscript_Figures.ipynb`: Evaluate the performance of the final MPNN (after hyperparameter tuning) on the dataset
3. `contributions_00.ipynb`: Generate per-atom contributions by running inference on masked molecules
4. `contributions_01.ipynb`: Analyzing and generating graphics of per-atom contributions

## Code
The `src` directory contains python files describing the deepchem model and for carrying out hyperparameter searches.
