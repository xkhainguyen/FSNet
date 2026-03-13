import torch
import torch.nn.functional as F
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


class MixtureOfExperts(nn.Module):
    """Mixture of Experts with MLP experts and a learned gating network.

    Args:
        input_dim:   Dimension of the input features.
        hidden_dim:  Hidden dimension for each expert MLP.
        output_dim:  Dimension of the output.
        num_experts: Number of expert MLPs (default 4).
        top_k:       Number of experts to activate per token.  0 or
                     ``num_experts`` → dense (soft) routing; otherwise
                     sparse top-K routing (default 2).
        num_layers:  Hidden layers inside each expert (passed to MLP).
        dropout:     Dropout rate inside each expert (passed to MLP).

    The gating network is a single ``Linear(input_dim, num_experts)``
    layer.  A load-balancing auxiliary loss (Switch-Transformer style) is
    computed on every forward pass and stored as ``self.aux_loss``; add
    ``moe_aux_loss_weight * model.aux_loss`` to the training loss to
    encourage uniform expert utilisation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k if (0 < top_k < num_experts) else num_experts  # 0 → dense
        self.sparse = (self.top_k < num_experts)

        # Gating network: maps input → expert logits
        self.gate = nn.Linear(input_dim, num_experts)

        # Expert pool
        self.experts = nn.ModuleList([
            MLP(input_dim, hidden_dim, output_dim,
                num_layers=num_layers, dropout=dropout)
            for _ in range(num_experts)
        ])

        # Populated by forward(); used externally for aux-loss bookkeeping
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional sparse top-K routing.

        Returns:
            Tensor of shape ``(B, output_dim)``.
        """
        B = x.shape[0]
        E = self.num_experts
        K = self.top_k

        gate_logits = self.gate(x)                          # (B, E)
        gate_soft   = F.softmax(gate_logits, dim=-1)        # (B, E)  – always computed for aux loss

        # ---- Load-balancing auxiliary loss (Switch-Transformer style) ----
        # L_aux = E * sum_i( mean_i^2 )  — minimised when routing is uniform (= 1.0)
        mean_gate = gate_soft.mean(dim=0)                   # (E,)
        self.aux_loss = E * (mean_gate ** 2).sum()

        # ---- Compute all expert outputs (B, E, out_dim) ----
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, out)

        if not self.sparse:
            # Dense routing: weighted sum over all experts
            # gate_soft: (B, E, 1) · expert_outs: (B, E, out) → (B, out)
            out = (expert_outs * gate_soft.unsqueeze(-1)).sum(dim=1)
        else:
            # Sparse top-K routing
            topk_logits, topk_idx = gate_logits.topk(K, dim=-1)       # (B, K)
            topk_weights = F.softmax(topk_logits, dim=-1)              # (B, K)  re-normalised

            # Gather the K chosen expert outputs: (B, K, out)
            out_dim = expert_outs.shape[-1]
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, out_dim)   # (B, K, out)
            selected = expert_outs.gather(1, idx_exp)                  # (B, K, out)

            # Weighted sum over selected experts
            out = (selected * topk_weights.unsqueeze(-1)).sum(dim=1)   # (B, out)

        return out