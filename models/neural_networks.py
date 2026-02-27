import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]

        for i in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout/(i+1)))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class EnsembleMLP(nn.Module):
    """Wraps M independent MLP members and averages their predictions."""

    def __init__(self, models):
        super().__init__()
        self.members = nn.ModuleList(models)

    def forward(self, x):
        preds = torch.stack([m(x) for m in self.members], dim=0)  # (M, B, out)
        return preds.mean(dim=0)  # (B, out)

    def forward_all(self, x):
        """Return per-member predictions for uncertainty estimation."""
        return torch.stack([m(x) for m in self.members], dim=0)  # (M, B, out)