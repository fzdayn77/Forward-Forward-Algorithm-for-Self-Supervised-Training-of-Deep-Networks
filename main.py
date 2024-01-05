import torch
import os
from torch.optim import Adam
from utils.prepare_data import get_data
from models.encoder import get_encoder
from models.simCLR import SimCLR
from lightly.loss import NTXentLoss
from utils.train import train_simCLR

# Device
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Device ===> {DEVICE}')
NUM_WORKERS = os.cpu_count()

# Hyperparameters
CONFIGS = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'temperature': 0.7,
    'num_hidden_layers': 4}

# Data preparation
train_set, train_loader, test_set, test_loader = get_data(num_workers=NUM_WORKERS, dataset_name="cifar10", batch_size=CONFIGS['batch_size'])

# Encoder
encoder = get_encoder(model_name="resnet34", num_layers=CONFIGS['num_hidden_layers'], 
                      lr=CONFIGS['learning_rate'], temperature=CONFIGS['temperature'])

# Model
model = SimCLR(encoder, n_features=512, projection=True, device=DEVICE)
model.to(DEVICE)

# Loss
criterion = NTXentLoss(temperature=CONFIGS['temperature'], memory_bank_size=0)

# Optimizer
optimizer = Adam(model.parameters(), lr=CONFIGS['learning_rate'])

# Executing
train_simCLR(model=model, num_epochs=CONFIGS['num_epochs'], train_loader=train_loader,
             loss_func=criterion, optimizer=optimizer, device=DEVICE)
