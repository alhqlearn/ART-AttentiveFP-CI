# ATT-AttentiveFP-CI

## Project summary

This repository contains the code and data for the study "Machine Learning Model for Catalytic Asymmetric Reactions of Simple Alkenes: From Model to Chemical Insights." The project utilizes a manually curated dataset of asymmetric transformations of alkenes (ATT: AlkeneTransformationTriad) from literature to train a deep learning model focused on predicting reaction outcomes, particularly enantioselectivity. These reactions are crucial for catalytic enantioselective transformations of alkenes, yielding important building blocks such as cyclopropanes, aziridines, and arylated alkenes. Using this dataset various machine learning (ML) models are developed using different featurization techniques, including one-hot encoding, molecular fingerprints, SMILES, and molecular graphs. We used Optuna for hyper-parameter tuning for these ML models.

## Data

There are 376 reactions in total, each varying by the type of reacting partner, including alkene, chiral ligand, and substrate. The 'ATT_ind.csv' file includes reaction examples with SMILES strings for each component along with the corresponding enantiomeric excess (ee). The 'ATT_30_splits.xlsx' file contains data for 30 different splits.

### Environmental Setup

```
conda env create -f environment.yml
conda activate ATT-AttentiveFP-CI
pip install dgl-cu110
pip install dgllife==0.2.8
pip install optuna
pip install rdkit
```
## Demo & Instructions for use
Notebook1 showcases the training of a deep neural network (DNN) model using fingerprint techniques and Optuna for hyperparameter tuning. 

Notebook2 illustrates the training of a DNN model using one-hot encoding and Optuna for hyperparameter tuning. 

Notebook3 presents the training of Random Forest, SVM, Decision Tree, and Gradient Boosting models using one-hot encoding and Optuna for hyperparameter tuning.

Notebook4 details the training of the AttentiveFP model using the AttentiveFPAtomFeaturizer, which includes one-hot encodings for atom type, degree, hybridization, formal charge, and other relevant properties. 

Notebook5 covers the training of the AttentiveFP-CI model, also utilizing the AttentiveFPAtomFeaturizer, but with a different approach to handling class imbalance in the loss function.
