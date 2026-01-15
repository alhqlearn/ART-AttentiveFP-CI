# ART-AttentiveFP-CI

## Project summary

This repository contains the code and data for the study "Machine Learning Model for Catalytic Asymmetric Reactions of Simple Alkenes: From Model to Chemical Insights." The project utilizes a manually curated dataset of asymmetric transformations of alkenes (ART: AlkeneReactionTriad) from literature to train a deep learning model focused on predicting reaction outcomes, particularly enantioselectivity. These reactions are crucial for catalytic enantioselective transformations of alkenes, yielding important building blocks such as cyclopropanes, aziridines, and arylated alkenes. Using this dataset various machine learning (ML) models are developed using different featurization techniques, including one-hot encoding, molecular fingerprints, SMILES, and molecular graphs. We used Optuna for hyper-parameter tuning for these ML models.

## Dataset

```ART dataset:``` There are 376 reactions in total, each varying by the type of reacting partner, including alkene, chiral ligand, and substrate. The 'ART_ind.csv' file includes reaction examples with SMILES strings for each component along with the corresponding enantiomeric excess (ee). The 'ART_30_splits.xlsx' file contains data for 30 different splits.

The execl files for the benchmark datasets, namely the asymmetric hydrogenation reaction (Asymmetric_hydrogenation_reaction.csv), the N, S-acetylation reaction (N,S_acetylation_reaction.xlsx), Buchwald–Hartwig amination high throughput dataset (BHA_HTE.csv), Buchwald–Hartwig amination low throughput dataset (BHA_LTE.csv), USPTO gram scale (USPTO_gram_train.csv, USPTO_gram_test.csv) are also provided in the main branch.

Moreover, the dataset containg potential Pubchem ligands and their predicted %ee values are shown in Potential_Pubchem_Ligands_Predicted_ee.xlsx file.


# ART Dataset (Alkene Reaction Triad)

The **ART dataset** is a curated collection of **376 catalytic asymmetric alkene reactions** specifically designed as a challenging benchmark for enantioselectivity prediction in asymmetric catalysis.

The dataset covers **three major reaction classes**:

- Cyclopropanation  
- Aziridination  
- Arylation  

Each reaction entry varies in:
- Alkene identity
- Chiral ligand structure
- Reaction substrate

This structural diversity makes ART particularly valuable for testing the generalization ability of machine learning models in predicting enantiomeric excess (%ee).

## Dataset Files

### Core ART Dataset

| File                        | Description                                                                                     | Format |
|-----------------------------|-------------------------------------------------------------------------------------------------|--------|
| `ART_ind.csv`               | Individual reaction entries including SMILES of all components and experimental %ee values     | CSV    |
| `ART_30_splits.xlsx`        | 30 predefined train–test splits for statistically robust and comparable model evaluation       | XLSX   |

These splits are strongly recommended for fair benchmarking and comparison between different machine learning approaches.

### Benchmark Reaction Datasets

The repository also includes several widely-used benchmark datasets from asymmetric catalysis and reaction prediction literature. These allow evaluation of model generalizability across different reaction classes, data sizes, and experimental regimes.

| Dataset                              | File(s)                                    | Description                                                                 |
|--------------------------------------|--------------------------------------------|-----------------------------------------------------------------------------|
| Asymmetric Hydrogenation             | `Asymmetric_hydrogenation_reaction.csv`    | Classic asymmetric hydrogenation reactions                                  |
| N,S-Acetylation                      | `N,S_acetylation_reaction.xlsx`            | N,S-acetylation reaction dataset                                            |
| Buchwald–Hartwig Amination (HTE)     | `BHA_HTE.csv`                              | High-throughput experimentation data                                        |
| Buchwald–Hartwig Amination (LTE)     | `BHA_LTE.csv`                              | Low-throughput (traditional) experimentation data                           |
| USPTO Gram-scale Reactions           | `USPTO_gram_train.csv`<br>`USPTO_gram_test.csv` | Gram-scale reaction dataset from USPTO (train/test split)                 |

These datasets are commonly used in the reaction prediction and asymmetric catalysis machine learning community.

### Predicted Ligand Screening Set

| File                                      | Description                                                                                   | Format |
|-------------------------------------------|-----------------------------------------------------------------------------------------------|--------|
| `Potential_Pubchem_Ligands_Predicted_ee.xlsx` | Candidate chiral ligands sourced from PubChem with model-predicted %ee values                | XLSX   |

**Important note:**  
This file is intended for **virtual screening**, hypothesis generation, and prioritization of synthesis candidates — **not** for direct experimental validation or performance benchmarking.

### Environmental Setup

```
conda env create -f environment.yml
conda activate ART-AttentiveFP-CI
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

``It is important to highlight that Optuna is utilized for hyperparameter tuning to find the most promising hyperparameter sets for all these ML models.``

## Acknowledgement
We would like to acknowledge the following works:

https://github.com/Sunojlab/Transfer_Learning_in_Catalysis

https://github.com/skinnider/low-data-generative-models

https://github.com/isayev/ReLeaSE

https://github.com/alhqlearn/REEXPLORE.git


## Citation
