import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import time

def train_simCLR(model: nn.Module, num_epochs: int, train_loader: DataLoader, loss_func, optimizer=None, device=None):
  print("Begin training ...")
  start_time = time.time()

  for epoch in range(num_epochs):
    # Putting the model in training mode
    model.train()
    total_loss = 0

    inner_tqdm = tqdm(train_loader, desc=f"Training | Epoch {epoch+1}/{num_epochs}", leave=True, position=0)

    for i, batch in enumerate(inner_tqdm):
      #inputs, labels = data
      x0, x1 = batch[0]
      x0 = x0.to(device)
      x1 = x1.to(device)

      optimizer.zero_grad()

      z0 = model(x0)
      z1 = model(x1)

      loss = loss_func(z0, z1)
      total_loss += loss.detach()

      loss.backward()
      optimizer.step()
        
    avg_loss = total_loss / len(train_loader)
    print(f"loss: {avg_loss:.5f}")

  # Total time
  elapsed = (time.time() - start_time) / 60
  print(f'Total training time: {elapsed:.2f} min')
  print("Training Done!\n")