import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import random
import math
import time

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../code/')
import util
from autoencoder import *
import autoencoder

import wandb
import random # for demo script
wandb.login()






data_path = '/data-ssd/sookim/data_full/transformer_data'
spec = 'airworthy' #'hover'#'interference'#'dist' #'mass' #'airworthy'
max_mass = 35. #(kg)

torch.manual_seed(0)
np.random.seed(0)
batch_size = 10 # 512
batch_size_val = 16 # 512
frac_train = 0.8
frac_val = 0.1

dataloader_tr, dataloader_val, dataloader_test, scale_1, scale_2 = autoencoder.prepare_sequence_data(data_path, spec, batch_size = batch_size ,batch_size_val = batch_size_val, frac_train = frac_train, frac_val = frac_val)

print(f'Training Data:   {dataloader_tr.dataset.y_train.shape[0]}')
print(f'Validation Data: {dataloader_val.dataset.y_train.shape[0]}')
print(f'Test Data:       {dataloader_test.dataset.y_train.shape[0]}')

### Set up model from the seq_to_spec_model.py
torch.manual_seed(0)
np.random.seed(0)

emsize = 200  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 8  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 20  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
D_out = 200
D = dataloader_tr.dataset.x_train.shape[-1]
"""
torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device="cuda"
else: 
    device = "cpu"
"""
device="cpu"
#torch.cuda.set_device(3)
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

model = TransformerAutoencoder( emsize, nhead, d_hid, nlayers, dropout, D, D_out).to(device)
##Trouble shooting
batch, train_data=next(enumerate(dataloader_tr))
data, targets, mask = train_data
print(data.size(), targets.size(), mask.size())
output= model(data.to(device), mask.to(device))
print(output.size())
### Set up the training routine
criterion = nn.MSELoss()
lr = .1  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = 0 #bptt
    for batch, train_data in enumerate(dataloader_tr):
        data, targets, mask = train_data
        output = model(data.to(device), mask.to(device))
        loss = criterion(output, data.to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss/num_batches

def evaluate(model: nn.Module) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    num_batches = 0
    with torch.no_grad():
        for batch, val_data in enumerate(dataloader_val):
            data, targets, mask = val_data
            output = model(data.to(device), mask.to(device))
            loss = criterion(output, data)
            num_batches += 1
            total_loss += loss.item()
    return total_loss/num_batches
### Train the model and save the best according to the validation data

best_loss = float('inf')
epochs = 100
best_model = None

loss_list = []
val_loss_list = []

run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    })
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    loss = train(model)
    val_loss = evaluate(model)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'loss {loss:5.4f} | '
          f'val loss {val_loss:5.4f}' )
    print('-' * 89)

    if val_loss < best_loss:
        best_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model.save(model.state_dict(), "./")
    loss_list.append(loss)
    val_loss_list.append(val_loss)


plt.plot(loss_list)
plt.plot(val_loss_list)
plt.grid()
plt.savefig("training_curve_autoencoder.png")
plt.close()

output_all =[]
i = 0
for x,y,m in dataloader_test:
    with torch.no_grad():
        print(x.shape)
        best_model.eval().to(device)
        output = best_model(x.to(device), m.to(device)).cpu()
        output_all.append(output)
        print("MSE btw input and output of autoencoder",criterion(x, output).item())
np.save("./output_mse.npy", np.asarray(output_all))
