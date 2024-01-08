import torch
import torch.nn as nn
from configs import CONFIGS

# Run: "$pip install lightly" to install lightly first
from lightly.loss import NTXentLoss


class FF_Layer(nn.Linear):
    def __init__(self, in_features, out_features, num_epochs, lr, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, dtype)
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        
        self.relu = torch.nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_func = NTXentLoss(temperature=CONFIGS['temperature'], memory_bank_size=0)
        #self.threshold = 2.0

    def forward(self, x):
        # Between layers'normalization
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_i, x_j):
        for e in range(self.num_epochs):
            loss = self.loss_func(x_i, x_j)
            self.optimizer.zero_grad()
            #loss.backward(create_graph=True)
            self.optimizer.step()
        return self.forward(x_i).detach(), self.forward(x_j).detach()


class FF_Net(nn.Module):
  def __init__(self, in_features, out_features, num_hidden_layers, num_epochs, lr, device=None):
    super().__init__()
    self.num_hidden_layers = num_hidden_layers
    self.num_epochs = num_epochs
    self.lr = lr
    self.device = device
    self.in_features = in_features
    self.out_features = out_features

    # Features extraction
    self.features = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=1),
        nn.BatchNorm2d(64),

        torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=1),
        nn.BatchNorm2d(128),

        torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=1),
        nn.BatchNorm2d(256),
    )

    # Average Pooling
    self.avgpool = torch.nn.AdaptiveAvgPool2d((2, 2))

    # Forward-Forward-Layers
    ff_layers = [
        FF_Layer(in_features=1024 if idx == 0 else 2000,
                 out_features=2000,
                 num_epochs=self.num_epochs,
                 lr=self.lr,
                 device=self.device) for idx in range(self.num_hidden_layers)
    ]
    self.ff_layers = ff_layers

  def train_net(self, x_i, x_j):
      x_1, x_2 = x_i, x_j
      x_1 = x_1.to(self.device)
      x_2 = x_2.to(self.device)

      x_1 = self.features(x_1)
      x_1 = self.avgpool(x_1)
      x_1 = torch.flatten(x_1, start_dim=1)

      x_2 = self.features(x_2)
      x_2 = self.avgpool(x_2)
      x_2 = torch.flatten(x_2, start_dim=1)

      # Pass the flattened features through the FF layers
      for _ , layer in enumerate(self.ff_layers):
          layer = layer.to(self.device)
          x_1, x_2 = layer.train(x_1, x_2)
      return x_1, x_2

  def forward(self, x_i, x_j):
      h_i, h_j = self.train_net(x_i, x_j)
      return h_i, h_j
    