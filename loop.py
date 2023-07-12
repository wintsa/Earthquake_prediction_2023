# System
import os

# Data processing
import numpy as np

# Results presentation
from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt

# NN related stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

# All hyperparameters are listed in model.py you can change them there
import train
from model import *


N_CELLS_HOR = 200
N_CELLS_VER = 250


celled_data = torch.load("Data/celled_data_"
                         + str(N_CELLS_HOR)
                         + "x"
                         + str(N_CELLS_VER))
# celled_data = celled_data[:350]
print (celled_data.shape)




DEVICE_ID = 0
# torch.cuda.set_device(DEVICE_ID)
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
print (DEVICE)

OBSERVED_DAYS = 64     # ~2 months
DAYS_TO_PREDICT_AFTER  = 10
DAYS_TO_PREDICT_BEFORE = 50
TESTING_DAYS = 1000

HEAVY_QUAKE_THRES = 3.5

freq_map = (celled_data>HEAVY_QUAKE_THRES).float().mean(dim=0)

class Dataset_RNN_Train(Dataset):
    def __init__(self, celled_data):
        self.data = celled_data[0:
                                (celled_data.shape[0] -
                                 TESTING_DAYS)]
        self.size = (self.data.shape[0] -
                     DAYS_TO_PREDICT_BEFORE)

        print('self.data :', self.data.shape)
        print('size      :', self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.data[idx],
                torch.sum(self.data[(idx +
                                     DAYS_TO_PREDICT_AFTER):
                                    (idx +
                                     DAYS_TO_PREDICT_BEFORE)] > HEAVY_QUAKE_THRES,
                          dim=0,
                          keepdim=True).squeeze(0) > 0)

dataset_train = Dataset_RNN_Train (celled_data)

dataloader_train = DataLoader(dataset_train,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0)

N_CYCLES = 10
QUEUE_LENGHT = 50
LEARNING_RATE = 0.0003
LR_DECAY = 10.
EARTHQUAKE_WEIGHT = 10000.


torch.cuda.empty_cache()

torch.autograd.set_detect_anomaly(True)

RNN_cell = LSTMCell(freq_map,
                    embedding_size    = EMB_SIZE,
                    hidden_state_size = HID_SIZE,
                    n_cells_hor       = N_CELLS_HOR,
                    n_cells_ver       = N_CELLS_VER,
                    device            = DEVICE)

torch.cuda.empty_cache()

train.train_RNN_full (RNN_cell,
                      DEVICE,
                      dataloader_train,
                      n_cycles=1,
                      learning_rate=LEARNING_RATE,
                      earthquake_weight=EARTHQUAKE_WEIGHT,
                      lr_decay=LR_DECAY)

torch.cuda.empty_cache()

train.train_RNN_part (RNN_cell,
                      DEVICE,
                      dataset_train,
                      n_cycles=N_CYCLES,
                      queue_lenght=QUEUE_LENGHT,
                      learning_rate=LEARNING_RATE,
                      earthquake_weight=EARTHQUAKE_WEIGHT,
                      lr_decay=LR_DECAY)

torch.cuda.empty_cache()

train.train_RNN_full (RNN_cell,
                      DEVICE,
                      dataloader_train,
                      n_cycles=1,
                      learning_rate=LEARNING_RATE,
                      earthquake_weight=EARTHQUAKE_WEIGHT,
                      lr_decay=LR_DECAY)

if not os.path.exists("Model"):
    os.mkdir("Model")
torch.save(RNN_cell.state_dict(), "Model/state_dict")