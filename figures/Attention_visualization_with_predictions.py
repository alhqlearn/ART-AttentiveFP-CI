
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from rdkit.Chem import rdDepictor, Draw

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import os 
import sys

import math

from torch.utils.data import DataLoader
from IPython.display import SVG, display
from PIL import Image
from io import BytesIO

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DCairo
import numpy as np
import dgl
from matplotlib import cm

import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
from dgllife.data import MoleculeCSVDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import optuna


def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks




def load_data(data,name):
    cache_file_path = f"{name}_dataset.bin"
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer= bond_featurizer,
                                 smiles_column='smiles',
                                 task_names=['ee'],init_mask=True,n_jobs=8,
                                 cache_file_path=cache_file_path
                                )
    return dataset


def data_loader_fn(dc_listings1, dc_listings2, dc_listings3):
    train_datasets = load_data(dc_listings1,'train')
    valid_datasets = load_data(dc_listings2,'valid')
    test_datasets = load_data(dc_listings3,'test')
    train_loader = DataLoader(train_datasets, batch_size=64,shuffle=False,
                              collate_fn=collate_molgraphs)
    valid_loader = DataLoader(valid_datasets,batch_size=64,shuffle=False,
                              collate_fn=collate_molgraphs)
    test_loader = DataLoader(test_datasets,batch_size=64,shuffle=False,
                              collate_fn=collate_molgraphs)
    return train_loader, valid_loader, test_loader, train_datasets, valid_datasets, test_datasets



def dataset_func(data_file):
    X =data_file['smiles']
    y=data_file['ee']
    
    #y = y.sample(frac=1, random_state=42).reset_index(drop=True)  # for Y scrambling
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, shuffle=False)

    print("Train Dataset: {}".format(X_train.shape))
    print("Val Dataset: {}".format(X_val.shape))
    print("Test Dataset: {}".format(X_test.shape))

    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    return df_train, df_val, df_test




atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')


data_file_path = 'divya_dataset_split_21.xlsx'



num_epoch = 300
data_file = pd.read_excel(data_file_path, sheet_name='fullcv_21')
dc_listings1, dc_listings2, dc_listings3 = dataset_func(data_file)
train_loader, valid_loader, test_loader, train_datasets, valid_datasets, test_datasets = data_loader_fn(dc_listings1, dc_listings2, dc_listings3)


# In[11]:


test_datasets[73]


# In[12]:


def compute_loss(model, prediction, labels, masks, loss_criterion):
    # Class imbalance loss
    class_imbalance_loss = torch.mean(
        torch.where(labels < 50, (labels - prediction) ** 2, 0.5 * (labels - prediction) ** 2)
    )

    # Penalty for predictions outside the valid range [0, 100]
    range_penalty = torch.mean(
        torch.where((prediction < 0) | (prediction > 99), (prediction - torch.clamp(prediction, 0, 99)) ** 2, 0)
    )



    # Total loss = 70% class imbalance loss + 30% range penalty
    #total_loss =  0.5*class_imbalance_loss +  range_penalty
    total_loss = class_imbalance_loss 

    return total_loss


# In[14]:


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

best_n_layers = 2  # Set a fixed value for the number of layers
best_graph_feat_size = 300  # Set a fixed value for the graph feature size
best_dropout_rate = 0.3
best_learning_rate = 0.001

best_n_layers = 2  # Set a fixed value for the number of layers
best_graph_feat_size = 394  # Set a fixed value for the graph feature size
best_dropout_rate = 0.31
best_learning_rate = 0.00031

atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')

model = model_zoo.AttentiveFPPredictor(node_feat_size=n_feats,
                                   edge_feat_size=e_feats,
                                   num_layers=best_n_layers,
                                   num_timesteps=1,
                                   graph_feat_size=best_graph_feat_size,
                                   n_tasks=1,
                                   dropout=best_dropout_rate
                                    )
model = model.to(device)
#Train
loss_criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=best_learning_rate, weight_decay = 0.000001) 
criterion = nn.MSELoss()


fn = 'model.pth'

best_n_layers = 2  # Set a fixed value for the number of layers
best_graph_feat_size = 300  # Set a fixed value for the graph feature size
best_dropout_rate = 0.3
best_learning_rate = 0.001

model = model_zoo.AttentiveFPPredictor(node_feat_size=n_feats,
                                   edge_feat_size=e_feats,
                                   num_layers=best_n_layers,
                                   num_timesteps=1,
                                   graph_feat_size=best_graph_feat_size,
                                   n_tasks=1,
                                   dropout=best_dropout_rate
                                    )

model.load_state_dict(torch.load(fn,map_location=torch.device('cpu'))) #'cuda:7'
model.to(device)

data_file_oob = 'Potential_Pubchem_Ligands.csv'
# Keep one empty column for ee in the above dataset

oob = pd.read_csv(data_file_oob)

# In[16]:


oob_datasets = load_data(oob,'oob')
oob_loader = DataLoader(oob_datasets, batch_size=64,shuffle=False,
                          collate_fn=collate_molgraphs)


# In[17]:


criterion = nn.MSELoss()
model.eval()
with torch.no_grad():
    labels_app = []
    test_prediction_app = []
    smiles_app = []
    for batch_id, batch_data in enumerate(oob_loader):
        smiles, bg, labels, masks = batch_data
        smiles_app.append(smiles)
        #print(smiles)
        bg = bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        test_prediction = model(bg, n_feats, e_feats)
        labels_app.extend(labels)
        test_prediction_app.extend(test_prediction)
    test_rmse = math.sqrt(criterion(torch.cat(test_prediction_app), torch.cat(labels_app)).item())
    test_r2 = r2_score(torch.cat(labels_app).detach().cpu().numpy(), torch.cat(test_prediction_app).detach().cpu().numpy())


print('All predicted ee : ', torch.cat(test_prediction_app).detach().cpu().numpy())
oob['Predicted_ee'] = torch.cat(test_prediction_app).detach().cpu().numpy()
oob.to_csv('Potential_Pubchem_Ligands_Predicted_ee.csv', index = None)
# In[19]:


def draw_similarity_maps(mol_id, dataset, timestep):
    smiles, g, label, _ = dataset[mol_id]
    #print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    g = dgl.batch([g])
    atom_feats, bond_feats = g.ndata.pop('hv'), g.edata.pop('he')
    preds, atom_weights1 = model(g, atom_feats, bond_feats, get_node_weight=True)
    assert timestep < len(atom_weights1)
    atom_weights1 = atom_weights1[timestep]
    atom_weights = atom_weights1.detach().cpu().numpy().flatten().tolist()
    nanMean = np.nanmean(atom_weights)
    #print(atom_weights-nanMean)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, atom_weights-nanMean,alpha=0.3, size=(400, 400))
    savepath = f'similarity_map_test_sample_{mol_id}_{preds.item()}.png'
    fig.savefig(savepath, bbox_inches='tight', dpi=300)
    return smiles,label,preds

for i in range(oob.shape[0]):
    draw_similarity_maps(i, oob_datasets, 0)
