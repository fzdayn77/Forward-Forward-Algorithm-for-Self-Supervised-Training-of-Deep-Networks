import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import time


def train_one_epoch(model, train_tqdm, loader_size, loss_func, optimizer, device):
    model.train(True)
    total_loss = 0
    train_losses = []
    
    for _ , batch in enumerate(train_tqdm):
      optimizer.zero_grad()
    
      (x_i, x_j), _ = batch
      x_i, x_j = x_i.to(device), x_j.to(device)

      _, _, z_i, z_j = model(x_i, x_j)

      loss = loss_func(z_i, z_j)
      loss.backward()
        
      optimizer.step()
      total_loss += loss.item()
    
    avg_loss = total_loss / loader_size
    train_losses.append(avg_loss)
    
    return avg_loss, train_losses


def test_one_epoch(model, test_loader, loss_func, device):
    model.eval()
    total_loss = 0
    validation_losses = []
    
    with torch.no_grad():
        for _ , batch in enumerate(test_loader):
          (x_i, x_j), _ = batch
          x_i, x_j = x_i.to(device), x_j.to(device)

          _, _, z_i, z_j = model(x_i, x_j)

          loss = loss_func(z_i, z_j)
          total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    validation_losses.append(avg_loss)
    
    return avg_loss, validation_losses


def compute_train_val_loss_BP(model: nn.Module, num_epochs: int, train_loader: DataLoader, test_loader: DataLoader, loss_func, optimizer=None, device=None):
  start_time = time.time()
  train_losses_list = []
  validation_losses_list = []
  
  for epoch in range(num_epochs):
    train_tqdm = tqdm(train_loader, desc=f"Training | Epoch {epoch+1}/{num_epochs}", leave=True, position=0)
    
    # Training
    avg_train_loss, train_losses = train_one_epoch(model, train_tqdm, 
                                             len(train_loader), loss_func, optimizer, device)
    train_losses_list.append([loss for loss in train_losses])
    
    # Validation
    avg_val_loss, validation_losses = test_one_epoch(model, test_loader, loss_func, device)
    validation_losses_list.append([loss for loss in validation_losses])

    # Logging training and validation losses
    print(f"Training loss: {avg_train_loss:.4f} || Validation loss: {avg_val_loss:.4f}")
    
  # Total time
  elapsed = (time.time() - start_time) / 60
  print(f'Total time: {elapsed:.2f} min')
  print("Done!")

  return train_losses_list, validation_losses_list