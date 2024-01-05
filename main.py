import torch
from utils.prepare_data import get_data
from models.encoder import get_encoder

# Device
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Device ===> {DEVICE}')

# Hyperparameters
CONFIGS = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'temperature': 0.7,
    'num_hidden_layers': 4}

# Data preparation
train_set, train_loader, test_set, test_loader = get_data(dataset_name="cifar10", batch_size=CONFIGS['batch_size'])

# Encoder
encoder = get_encoder(model_name="forward-forward",
                      num_layers=CONFIGS['num_hidden_layers'],
                      lr=CONFIGS['learning_rate'],
                      temperature=CONFIGS['temperature'],
                      device=DEVICE)

# Loss
# TODO

# Optimizer
# TODO

# Executing
# TODO