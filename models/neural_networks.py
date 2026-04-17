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
                layers.append(nn.Dropout(p=dropout / (i + 1)))

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class _ContextFeatureBase(nn.Module):
    """Shared utilities for context-augmented models."""

    def __init__(
        self,
        *,
        problem_type: str,
        problem_name: str,
        L: torch.Tensor,
        U: torch.Tensor,
        A: torch.Tensor,
        G: torch.Tensor,
        h: torch.Tensor,
        Q: torch.Tensor,
        p: torch.Tensor,
        c: float | torch.Tensor,
        H: torch.Tensor | None = None,
        C: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        num_context_points: int = 16,
        seed: int = 2025,
    ):
        super().__init__()
        self.problem_type = problem_type
        self.problem_name = problem_name
        self.num_context_points = int(num_context_points)

        self.register_buffer("L", L.detach().clone())
        self.register_buffer("U", U.detach().clone())
        self.register_buffer("A", A.detach().clone())
        self.register_buffer("G", G.detach().clone())
        self.register_buffer("h", h.detach().clone())
        self.register_buffer("Q", Q.detach().clone())
        self.register_buffer("p", p.detach().clone())
        self.register_buffer("c", torch.as_tensor(c, dtype=L.dtype, device=L.device))

        if H is not None:
            self.register_buffer("H", H.detach().clone())
        else:
            self.H = None
        if C is not None:
            self.register_buffer("C", C.detach().clone())
        else:
            self.C = None
        if d is not None:
            self.register_buffer("d", d.detach().clone())
        else:
            self.d = None

        self.eq_dim = int(self.A.shape[0])
        self.ineq_dim = int(self.G.shape[0] + 2 * self.L.numel())
        self.point_feature_dim = 1 + self.eq_dim + self.ineq_dim

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        span = (self.U - self.L).detach().cpu().unsqueeze(0)
        lower = self.L.detach().cpu().unsqueeze(0)
        y_ref = lower + torch.rand(
            self.num_context_points,
            self.L.numel(),
            generator=gen,
            dtype=self.L.dtype,
        ) * span
        self.register_buffer("context_y_ref", y_ref.to(device=self.L.device, dtype=self.L.dtype))

    def _obj_fn(self, Y: torch.Tensor) -> torch.Tensor:
        quad = (0.5 * (Y @ self.Q) * Y).sum(dim=1)
        if self.problem_type == "convex":
            return quad + (self.p * Y).sum(dim=1) + self.c
        if self.problem_type == "nonconvex":
            return quad + (self.p * torch.sin(Y)).sum(dim=1) + self.c
        if self.problem_type == "nonsmooth_nonconvex":
            return quad + (self.p * torch.sin(Y)).sum(dim=1) + 0.1 * torch.norm(Y, dim=1) + self.c
        raise ValueError(f"Unknown problem_type: {self.problem_type}")

    def _eq_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return Y @ self.A.T - X

    def _ineq_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if self.problem_name == "qp":
            if self.problem_type == "convex":
                res = Y @ self.G.T - self.h.view(1, -1)
            else:
                res = torch.sin(Y) @ self.G.T - self.h.view(1, -1) * torch.cos(X)
            return torch.clamp(torch.cat([res, self.L - Y, Y - self.U], dim=1), 0)

        if self.problem_name == "qcqp":
            q = torch.matmul(self.H, Y.T).permute(2, 0, 1)
            q = (q * Y.view(Y.shape[0], 1, -1)).sum(-1)
            if self.problem_type == "convex":
                res = 0.5 * q + torch.matmul(Y, self.G.T) - self.h
            else:
                res = 0.5 * q + torch.matmul(torch.cos(Y), self.G.T) - self.h
            return torch.clamp(torch.cat([res, self.L - Y, Y - self.U], dim=1), 0)

        if self.problem_name == "socp":
            if self.problem_type == "convex":
                q = torch.norm(torch.matmul(self.G, Y.T).permute(2, 0, 1) + self.h.unsqueeze(0), dim=-1, p=2)
            else:
                q = torch.norm(torch.matmul(self.G, torch.cos(Y).T).permute(2, 0, 1) + self.h.unsqueeze(0), dim=-1, p=2)
            p = torch.matmul(Y, self.C.T) + self.d
            res = q - p
            return torch.clamp(torch.cat([res, self.L - Y, Y - self.U], dim=1), 0)

        raise ValueError(f"Unknown problem_name: {self.problem_name}")

    def _compute_point_features(self, X: torch.Tensor) -> torch.Tensor:
        batch_size = X.shape[0]
        num_refs = self.context_y_ref.shape[0]
        x_rep = X.unsqueeze(1).expand(-1, num_refs, -1).reshape(batch_size * num_refs, -1)
        y_rep = self.context_y_ref.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * num_refs, -1)

        obj = self._obj_fn(y_rep).view(batch_size, num_refs, 1)
        eq = self._eq_resid(x_rep, y_rep).view(batch_size, num_refs, -1)
        ineq = self._ineq_resid(x_rep, y_rep).view(batch_size, num_refs, -1)
        return torch.cat([obj, eq, ineq], dim=2)


class SampledContextMLPv1(_ContextFeatureBase):
    """MLP that augments ``x`` with flattened full residual-vector features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        problem_type: str,
        problem_name: str,
        L: torch.Tensor,
        U: torch.Tensor,
        A: torch.Tensor,
        G: torch.Tensor,
        h: torch.Tensor,
        Q: torch.Tensor,
        p: torch.Tensor,
        c: float | torch.Tensor,
        H: torch.Tensor | None = None,
        C: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        num_context_points: int = 16,
        seed: int = 2025,
        context_normalize: bool = True,
        context_eps: float = 1e-8,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(
            problem_type=problem_type,
            problem_name=problem_name,
            L=L,
            U=U,
            A=A,
            G=G,
            h=h,
            Q=Q,
            p=p,
            c=c,
            H=H,
            C=C,
            d=d,
            num_context_points=num_context_points,
            seed=seed,
        )
        self.context_normalize = bool(context_normalize)
        self.context_eps = float(context_eps)
        self.context_feature_dim = self.num_context_points * self.point_feature_dim

        self.register_buffer("context_mean", torch.zeros(self.context_feature_dim, dtype=self.L.dtype, device=self.L.device))
        self.register_buffer("context_std", torch.ones(self.context_feature_dim, dtype=self.L.dtype, device=self.L.device))

        self.mlp = MLP(
            input_dim + self.context_feature_dim,
            hidden_dim,
            output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def _compute_context_features(self, X: torch.Tensor) -> torch.Tensor:
        flat = self._compute_point_features(X).reshape(X.shape[0], -1)
        if self.context_normalize:
            flat = (flat - self.context_mean.unsqueeze(0)) / self.context_std.unsqueeze(0)
        return flat

    @torch.no_grad()
    def fit_context_stats(self, X_train: torch.Tensor, batch_size: int = 256) -> None:
        if not self.context_normalize:
            self.context_mean.zero_()
            self.context_std.fill_(1.0)
            return

        device = self.context_y_ref.device
        total = 0
        feat_sum = torch.zeros_like(self.context_mean)
        feat_sq_sum = torch.zeros_like(self.context_mean)

        for start in range(0, X_train.shape[0], batch_size):
            xb = X_train[start:start + batch_size].to(device=device, dtype=self.context_y_ref.dtype)
            feats = self._compute_context_features(xb)
            feat_sum += feats.sum(dim=0)
            feat_sq_sum += feats.square().sum(dim=0)
            total += feats.shape[0]

        total = max(total, 1)
        mean = feat_sum / total
        var = torch.clamp(feat_sq_sum / total - mean.square(), min=self.context_eps)
        self.context_mean.copy_(mean)
        self.context_std.copy_(torch.sqrt(var))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self._compute_context_features(x)
        return self.mlp(torch.cat([x, context], dim=1))


class SampledContextMLPv2(_ContextFeatureBase):
    """Context model with per-point encoding and mean pooling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        problem_type: str,
        problem_name: str,
        L: torch.Tensor,
        U: torch.Tensor,
        A: torch.Tensor,
        G: torch.Tensor,
        h: torch.Tensor,
        Q: torch.Tensor,
        p: torch.Tensor,
        c: float | torch.Tensor,
        H: torch.Tensor | None = None,
        C: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        num_context_points: int = 4,
        seed: int = 2025,
        context_normalize: bool = True,
        context_eps: float = 1e-8,
        context_encoder_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(
            problem_type=problem_type,
            problem_name=problem_name,
            L=L,
            U=U,
            A=A,
            G=G,
            h=h,
            Q=Q,
            p=p,
            c=c,
            H=H,
            C=C,
            d=d,
            num_context_points=num_context_points,
            seed=seed,
        )
        self.context_normalize = bool(context_normalize)
        self.context_eps = float(context_eps)
        self.context_encoder_dim = int(context_encoder_dim)
        self.context_feature_dim = self.context_encoder_dim

        self.register_buffer("context_mean", torch.zeros(self.point_feature_dim, dtype=self.L.dtype, device=self.L.device))
        self.register_buffer("context_std", torch.ones(self.point_feature_dim, dtype=self.L.dtype, device=self.L.device))

        self.point_encoder = nn.Sequential(
            nn.Linear(self.point_feature_dim, self.context_encoder_dim),
            nn.SiLU(),
            nn.Linear(self.context_encoder_dim, self.context_encoder_dim),
            nn.SiLU(),
        )
        self.mlp = MLP(
            input_dim + self.context_encoder_dim,
            hidden_dim,
            output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def _normalize_point_features(self, feats: torch.Tensor) -> torch.Tensor:
        if not self.context_normalize:
            return feats
        return (feats - self.context_mean.view(1, 1, -1)) / self.context_std.view(1, 1, -1)

    @torch.no_grad()
    def fit_context_stats(self, X_train: torch.Tensor, batch_size: int = 256) -> None:
        if not self.context_normalize:
            self.context_mean.zero_()
            self.context_std.fill_(1.0)
            return

        device = self.context_y_ref.device
        total = 0
        feat_sum = torch.zeros_like(self.context_mean)
        feat_sq_sum = torch.zeros_like(self.context_mean)

        for start in range(0, X_train.shape[0], batch_size):
            xb = X_train[start:start + batch_size].to(device=device, dtype=self.context_y_ref.dtype)
            feats = self._compute_point_features(xb).reshape(-1, self.point_feature_dim)
            feat_sum += feats.sum(dim=0)
            feat_sq_sum += feats.square().sum(dim=0)
            total += feats.shape[0]

        total = max(total, 1)
        mean = feat_sum / total
        var = torch.clamp(feat_sq_sum / total - mean.square(), min=self.context_eps)
        self.context_mean.copy_(mean)
        self.context_std.copy_(torch.sqrt(var))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        point_feats = self._normalize_point_features(self._compute_point_features(x))
        batch_size, num_refs, _ = point_feats.shape
        encoded = self.point_encoder(point_feats.reshape(batch_size * num_refs, -1))
        pooled = encoded.view(batch_size, num_refs, -1).mean(dim=1)
        return self.mlp(torch.cat([x, pooled], dim=1))


class LocalContextMLPv1(_ContextFeatureBase):
    """Two-stage model using local structure around a coarse prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        problem_type: str,
        problem_name: str,
        L: torch.Tensor,
        U: torch.Tensor,
        A: torch.Tensor,
        G: torch.Tensor,
        h: torch.Tensor,
        Q: torch.Tensor,
        p: torch.Tensor,
        c: float | torch.Tensor,
        H: torch.Tensor | None = None,
        C: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(
            problem_type=problem_type,
            problem_name=problem_name,
            L=L,
            U=U,
            A=A,
            G=G,
            h=h,
            Q=Q,
            p=p,
            c=c,
            H=H,
            C=C,
            d=d,
            num_context_points=1,
            seed=0,
        )
        if output_dim != self.L.numel():
            raise ValueError("LocalContextMLPv1 currently supports only full-output methods")

        self.local_feature_dim = self.point_feature_dim
        self.coarse_mlp = MLP(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.local_norm = nn.LayerNorm(self.local_feature_dim)
        self.refine_mlp = MLP(
            input_dim + output_dim + self.local_feature_dim,
            hidden_dim,
            output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def _scale_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * (self.U.view(1, -1) - self.L.view(1, -1)) + self.L.view(1, -1)

    def _compute_local_features(self, x: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
        y_scaled = self._scale_output(y_norm)
        obj = self._obj_fn(y_scaled).view(x.shape[0], 1)
        eq = self._eq_resid(x, y_scaled)
        ineq = self._ineq_resid(x, y_scaled)
        local = torch.cat([obj, eq, ineq], dim=1)
        return self.local_norm(local)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.coarse_mlp(x)
        local = self._compute_local_features(x, y0)
        return self.refine_mlp(torch.cat([x, y0, local], dim=1))


class LocalContextMLPv2(_ContextFeatureBase):
    """Residual local-refinement model with accessible coarse prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        problem_type: str,
        problem_name: str,
        L: torch.Tensor,
        U: torch.Tensor,
        A: torch.Tensor,
        G: torch.Tensor,
        h: torch.Tensor,
        Q: torch.Tensor,
        p: torch.Tensor,
        c: float | torch.Tensor,
        H: torch.Tensor | None = None,
        C: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        local_delta_scale: float = 0.2,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(
            problem_type=problem_type,
            problem_name=problem_name,
            L=L,
            U=U,
            A=A,
            G=G,
            h=h,
            Q=Q,
            p=p,
            c=c,
            H=H,
            C=C,
            d=d,
            num_context_points=1,
            seed=0,
        )
        if output_dim != self.L.numel():
            raise ValueError("LocalContextMLPv2 currently supports only full-output methods")

        self.local_feature_dim = self.point_feature_dim
        self.local_delta_scale = float(local_delta_scale)
        self.coarse_mlp = MLP(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.local_norm = nn.LayerNorm(self.local_feature_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(input_dim + output_dim + self.local_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        self.last_coarse_prediction: torch.Tensor | None = None
        self.last_refined_prediction: torch.Tensor | None = None

    def _scale_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * (self.U.view(1, -1) - self.L.view(1, -1)) + self.L.view(1, -1)

    def _compute_local_features(self, x: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
        y_scaled = self._scale_output(y_norm)
        obj = self._obj_fn(y_scaled).view(x.shape[0], 1)
        eq = self._eq_resid(x, y_scaled)
        ineq = self._ineq_resid(x, y_scaled)
        local = torch.cat([obj, eq, ineq], dim=1)
        return self.local_norm(local)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.coarse_mlp(x)
        local = self._compute_local_features(x, y0)
        delta = self.delta_head(torch.cat([x, y0, local], dim=1))
        y = torch.clamp(y0 + self.local_delta_scale * delta, 0.0, 1.0)
        self.last_coarse_prediction = y0
        self.last_refined_prediction = y
        return y


class EnsembleMLP(nn.Module):
    """Wraps M independent MLP members and averages their predictions."""

    def __init__(self, models):
        super().__init__()
        self.members = nn.ModuleList(models)

    def forward(self, x):
        preds = torch.stack([m(x) for m in self.members], dim=0)
        return preds.mean(dim=0)

    def forward_all(self, x):
        return torch.stack([m(x) for m in self.members], dim=0)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts with MLP experts and a learned gating network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        gate_temperature: float = 1.0,
        gate_noise_std: float = 0.0,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k if (0 < top_k < num_experts) else num_experts
        self.sparse = self.top_k < num_experts
        self.gate_temperature = gate_temperature
        self.gate_noise_std = gate_noise_std
        self.force_dense = False

        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout)
            for _ in range(num_experts)
        ])

        self.aux_loss: torch.Tensor = torch.tensor(0.0)
        self.gate_entropy: torch.Tensor = torch.tensor(0.0)
        self.gate_max_prob: torch.Tensor = torch.tensor(0.0)

    def set_routing(self, force_dense: bool = None, gate_temperature: float = None, gate_noise_std: float = None):
        if force_dense is not None:
            self.force_dense = bool(force_dense)
        if gate_temperature is not None:
            self.gate_temperature = max(float(gate_temperature), 1e-3)
        if gate_noise_std is not None:
            self.gate_noise_std = max(float(gate_noise_std), 0.0)

    def _compute_gates(self, x: torch.Tensor):
        gate_logits = self.gate(x)
        if self.training and self.gate_noise_std > 0:
            gate_logits = gate_logits + self.gate_noise_std * torch.randn_like(gate_logits)
        gate_logits = gate_logits / max(self.gate_temperature, 1e-3)
        gate_soft = F.softmax(gate_logits, dim=-1)

        mean_gate = gate_soft.mean(dim=0)
        self.aux_loss = self.num_experts * (mean_gate ** 2).sum()
        self.gate_entropy = (-gate_soft * torch.log(gate_soft + 1e-12)).sum(dim=-1).mean()
        self.gate_max_prob = gate_soft.max(dim=-1).values.mean()
        return gate_logits, gate_soft

    def forward_candidates(self, x: torch.Tensor, candidate_top_k: int = None):
        gate_logits, gate_soft = self._compute_gates(x)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)

        if candidate_top_k is None:
            effective_top_k = self.top_k
        else:
            effective_top_k = int(candidate_top_k)
            effective_top_k = max(1, min(effective_top_k, self.num_experts))

        use_sparse = (effective_top_k < self.num_experts) and (not self.force_dense)
        if not use_sparse:
            candidates = expert_outs
            topk_weights = gate_soft
            batch_size = x.shape[0]
            topk_idx = torch.arange(self.num_experts, device=x.device).unsqueeze(0).expand(batch_size, -1)
        else:
            topk_logits, topk_idx = gate_logits.topk(effective_top_k, dim=-1)
            topk_weights = F.softmax(topk_logits, dim=-1)

            out_dim = expert_outs.shape[-1]
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, out_dim)
            candidates = expert_outs.gather(1, idx_exp)

        out = (candidates * topk_weights.unsqueeze(-1)).sum(dim=1)
        return out, candidates, topk_idx, topk_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _, _ = self.forward_candidates(x)
        return out
