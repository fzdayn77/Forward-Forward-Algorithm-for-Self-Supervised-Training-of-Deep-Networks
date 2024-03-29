import time
import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader


def train_one_epoch(model, train_tqdm, loader_size, loss_func, optimizer, projection, device):
    model.train(True)
    total_loss = 0
    train_losses = []
    
    for _ , batch in enumerate(train_tqdm):
      optimizer.zero_grad()
      (x_i, x_j), _ = batch
      x_i, x_j = x_i.to(device), x_j.to(device)

      # When using a projection head
      if projection:
        _, _, z_i, z_j = model(x_i, x_j)
      else:
        z_i, z_j = model(x_i, x_j)

      loss = loss_func(z_i, z_j)
      loss.backward()
        
      optimizer.step()
      total_loss += loss.item()
    
    avg_loss = total_loss / loader_size
    train_losses.append(avg_loss)
    
    return avg_loss, train_losses


def test_one_epoch(model, test_loader, loss_func, projection, device):
    model.eval()
    total_loss = 0
    validation_losses = []
    
    with torch.no_grad():
        for _ , batch in enumerate(test_loader):
          (x_i, x_j), _ = batch
          x_i, x_j = x_i.to(device), x_j.to(device)

          # When using a projection head
          if projection:
            _, _, z_i, z_j = model(x_i, x_j)
          else:
            z_i, z_j = model(x_i, x_j)

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


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    
    for idx, batch in enumerate(loader):
        (x, _), y = batch
        x = x.to(device)
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)
        h = h.detach()
        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if idx % 30 == 0:
            print(f"Step [{idx}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_vectors(simclr_model, train_loader, test_loader, device):
    train_features, train_labels = inference(train_loader, simclr_model, device)
    test_features, test_labels = inference(test_loader, simclr_model, device)
    return train_features, train_labels, test_features, test_labels


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train_set = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False
    )

    test_set = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train_logistic_one_epoch(train_loader, logistic_model, criterion, optimizer, device):
    loss_epoch = 0
    accuracy_epoch = 0
    
    for _, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        x, y = batch
        x, y = x.to(device), y.to(device)

        output = logistic_model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        
        loss.backward()
        optimizer.step()
        
        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def test_logistic_one_epoch(test_loader, logistic_model, criterion, device):
    loss_epoch = 0
    accuracy_epoch = 0
    
    logistic_model.eval()
    for _, batch in enumerate(test_loader):
        logistic_model.zero_grad()
        x, y = batch
        x, y = x.to(device), y.to(device)

        output = logistic_model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        
        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch