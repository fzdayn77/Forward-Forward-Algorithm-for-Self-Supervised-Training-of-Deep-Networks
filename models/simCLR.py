import torch.nn as nn
from models.projection_head import Projection_Head

class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, n_features: int, projection: bool=True, device=None):
        super().__init__()
        self.encoder = encoder
        self.n_features = n_features
        self.projection = projection
        self.device = device

        if self.projection:
          self.projection_head = Projection_Head(input_dim=n_features, hidden_dim=n_features, 
                                                 output_dim=128, device=self.device)
          print("Projection Head ===> Enabled")
        else:
          print("Projection Head ===> Disabled")

    def forward(self, x):
        # x -> h
        h = self.encoder(x).flatten(start_dim=1)

        if self.projection:
          # h -> z
          z = self.projection_head(h)
          return z
        
        return h