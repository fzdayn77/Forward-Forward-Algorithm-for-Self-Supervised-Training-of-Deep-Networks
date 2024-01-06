import torch
import os
from torch.optim import Adam
from utils.prepare_data import get_data
from models.encoder import get_encoder
from models.simCLR import SimCLR
from lightly.loss import NTXentLoss
from utils.help_functions import compute_train_val_loss_BP

# Device
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Device ===> {DEVICE}')
NUM_WORKERS = os.cpu_count()

# Hyperparameters
CONFIGS = {
    'batch_size': 256,
    'learning_rate': 0.01,
    'num_epochs': 100,
    'temperature': 0.07,
    'num_hidden_layers': 4,
    'projection_dims': 128}

# Data preparation
train_set, train_loader, test_set, test_loader = get_data(num_workers=NUM_WORKERS, 
                                                          dataset_name="cifar10", batch_size=CONFIGS['batch_size'])

# Encoder
encoder = get_encoder(model_name="resnet18", num_layers=CONFIGS['num_hidden_layers'],
                      lr=CONFIGS['learning_rate'], temperature=CONFIGS['temperature'])
n_features = encoder.fc.in_features


# Model
simclr_model = SimCLR(encoder=encoder, projection_dim=CONFIGS['projection_dims'], 
                      n_features=n_features, projection=True)
simclr_model = simclr_model.to(DEVICE)

# Loss
criterion = NTXentLoss(temperature=CONFIGS['temperature'], memory_bank_size=0)

# Optimizer
optimizer = Adam(simclr_model.parameters(), lr=CONFIGS['learning_rate'])

# Executing
train_losses_list, validation_losses_list = compute_train_val_loss_BP(model=simclr_model, num_epochs=CONFIGS['num_epochs'], 
            train_loader=train_loader, loss_func=criterion, optimizer=optimizer, device=DEVICE)
