import torch.nn as nn
from models.identity import Identity

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim: int, n_features: int, projection=True):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.n_features = n_features
        self.projection = projection

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()
        
        if self.projection:
            # We use a MLP with one hidden layer
            self.projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_features, projection_dim, bias=False),
            )
            print("Projection Head ===> Enabled")
        else:
            print("Projection Head ===> Disabled")

    def forward(self, x_i, x_j):
        # x -> h
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        if self.projection:
            # h -> z
            z_i = self.projector(h_i)
            z_j = self.projector(h_j)  
            return h_i, h_j, z_i, z_j
        
        return h_i, h_j