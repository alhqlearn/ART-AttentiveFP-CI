# ART-AttentiveFP-CI

## Project summary

This repository contains the code and data for the study "Machine Learning Model for Catalytic Asymmetric Reactions of Simple Alkenes: From Model to Chemical Insights." The project utilizes a manually curated dataset of asymmetric transformations of alkenes (ART: AlkeneReactionTriad) from literature to train a deep learning model focused on predicting reaction outcomes, particularly enantioselectivity. These reactions are crucial for catalytic enantioselective transformations of alkenes, yielding important building blocks such as cyclopropanes, aziridines, and arylated alkenes. Using this dataset various machine learning (ML) models are developed using different featurization techniques, including one-hot encoding, molecular fingerprints, SMILES, and molecular graphs. We used Optuna for hyper-parameter tuning for these ML models.

## Dataset
### ART Dataset (Alkene Reaction Triad)

The **ART dataset** comprises 376 catalytic asymmetric alkene reactions, covering three reaction classes:
1. Cyclopropanation
2. Aziridination
3. Arylation

Each reaction varies in the identity of the alkene, chiral ligand, and reaction substrate, forming a comprehensive benchmark for enantioselectivity prediction.

### ART Core Data
| Filename | Description |
| :--- | :--- |
| **`ART_ind.csv`** | Contains individual reaction entries with SMILES representations of all reaction components and the corresponding experimentally reported enantiomeric excess (%ee) values. |
| **`ART_30_splits.xlsx`** | Provides 30 predefined train–test splits of the ART dataset to enable statistically robust model evaluation and fair comparison across different machine learning methods. |

---

### Benchmark Reaction Datasets

In addition to the ART dataset, several widely used benchmark reaction datasets are included in the main branch of this repository. These datasets are used to evaluate the generalizability of the proposed model across diverse reaction classes and data regimes.

### Asymmetric Hydrogenation
*   **`Asymmetric_hydrogenation_reaction.csv`**: Benchmark for asymmetric hydrogenation reactions.

### N,S-Acetylation Reaction
*   **`N,S_acetylation_reaction.xlsx`**: Data covering N,S-acetylation reactions.

### Buchwald–Hartwig Amination (BHA)
Datasets covering both High-Throughput (HTE) and Low-Throughput (LTE) experiment regimes:
*   **`BHA_HTE.csv`**: High-throughput experiments.
*   **`BHA_LTE.csv`**: Manually created LTE datapoints from HTE.

### USPTO Gram-Scale Reaction Dataset
A large-scale dataset split for training and evaluation:
*   **`USPTO_gram_train.csv`**: Training set.
*   **`USPTO_gram_test.csv`**: Test set.

---

## Predicted Ligand Dataset

*   **`Potential_Pubchem_Ligands_Predicted_ee.xlsx`**
    
    Contains candidate ligands sourced from PubChem along with their model-predicted enantiomeric excess (%ee) values. 

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

Notebook5 shows the creation of LTE dataset from HTE.
``It is important to highlight that Optuna is utilized for hyperparameter tuning to find the most promising hyperparameter sets for all these ML models.``

## Acknowledgement
We would like to acknowledge the following works:
https://github.com/OpenDrugAI/AttentiveFP.git

## Citation
