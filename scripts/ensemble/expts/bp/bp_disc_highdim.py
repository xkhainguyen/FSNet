"""Does the multimodal (disconnected-ball) perturbation gain survive in higher dim?

Tests the confound I flagged on the 2D result: in 2D, K=100 random perturbations
cover the space easily, so the "ball-switching" gain may be 2D-inflated random
multi-start. In R^d the same K covers exponentially less volume, so if the gain
is really just random search it should COLLAPSE as d grows; if it tracks the
NN's intrinsic wrong-ball rate (and best_merit still recovers), it's more robust.

Self-contained d-dimensional version (BP's Disconnected_Ball is hardcoded 2D):
  feasible set = ⋃_{i=1..n_ball} Ball(center_i ∈ R^d, radius_i),  min wᵀx.
Repair = radial bisection from the nearest ball center (fully converged).
Trained to convergence with cosine LR decay (the lesson from the 2D run).
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ineq_resid(c, x, n_ball, d):
    """min-over-balls signed distance, clamped to >=0. c:(B, n_ball*(d+1)) x:(B,d)"""
    B = x.shape[0]
    centers = c[:, : n_ball * d].view(B, n_ball, d)
    radii = c[:, n_ball * d:].view(B, n_ball)
    dist = torch.norm(x.view(B, 1, d) - centers, dim=-1) - radii  # (B, n_ball)
    return torch.clamp(dist.min(dim=1)[0], min=0.0)  # (B,)


def nearest_ball(c, x, n_ball, d):
    B = x.shape[0]
    centers = c[:, : n_ball * d].view(B, n_ball, d)
    idx = torch.norm(x.view(B, 1, d) - centers, dim=-1).argmin(dim=1)
    ar = torch.arange(B, device=c.device)
    return idx, centers[ar, idx]


def obj_fn(w, x):
    return (w * x).sum(dim=1)


def brute_force_opt(c, w, n_ball, d):
    B = c.shape[0]
    centers = c[:, : n_ball * d].view(B, n_ball, d)
    radii = c[:, n_ball * d:].view(B, n_ball)
    wn = w / w.norm(dim=1, keepdim=True)
    cand = centers - radii.unsqueeze(-1) * wn.unsqueeze(1)  # (B, n_ball, d)
    vals = (cand * wn.unsqueeze(1)).sum(dim=-1)             # (B, n_ball)
    best = vals.argmin(dim=1)
    ar = torch.arange(B, device=c.device)
    return obj_fn(w, cand[ar, best]), best


def radial_bisect(c, ip, cand, n_ball, d, n_steps=60, bis=0.9, eps=1e-7):
    B = c.shape[0]
    lo = torch.zeros(B, 1, device=c.device); hi = torch.ones(B, 1, device=c.device)
    for _ in range(n_steps):
        a = (1 - bis) * lo + bis * hi
        xt = a * (cand - ip) + ip
        fe = (ineq_resid(c, xt, n_ball, d) < eps).view(B, 1)
        lo = torch.where(fe, a, lo); hi = torch.where(~fe, a, hi)
        if (hi - lo).max() < 1e-12:
            break
    return lo * (cand - ip) + ip


class Solver(nn.Module):
    def __init__(self, cin, out, h=128, nl=4):
        super().__init__()
        layers = [nn.Linear(cin, h), nn.ReLU()]
        for _ in range(nl - 1):
            layers += [nn.Linear(h, h), nn.ReLU()]
        layers += [nn.Linear(h, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, c, w):
        return self.net(torch.cat([c, w], dim=1))


def sample(n, n_ball, d, rng):
    cdim = n_ball * (d + 1)
    c = np.empty((n, cdim))
    c[:, : n_ball * d] = rng.uniform(-1, 1, (n, n_ball * d))   # centers in [-1,1]^d
    c[:, n_ball * d:] = rng.uniform(0.5, 0.7, (n, n_ball))     # radii
    w = rng.normal(size=(n, d)); w /= np.linalg.norm(w, axis=1, keepdims=True)
    return torch.tensor(c, device=DEVICE), torch.tensor(w, device=DEVICE)


def train(n_ball, d, iters, h, nl, seed, lr0=1e-3):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    cdim = n_ball * (d + 1)
    model = Solver(cdim + d, d, h, nl).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr0 * 0.01)
    for it in range(iters):
        c, w = sample(256, n_ball, d, rng)
        x = model(c, w)
        lam = 1.0 + 99.0 * min(1.0, it / (0.3 * iters))
        loss = (obj_fn(w, x) + lam * ineq_resid(c, x, n_ball, d) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def eval_dim(model, n_ball, d, Ct, Wt, vopt, bb, Ks=(1, 100), epss=(0.1, 0.3, 0.5, 1.0)):
    model.eval()
    with torch.no_grad():
        cand0 = model(Ct, Wt)
    B = Ct.shape[0]; ar = torch.arange(B, device=DEVICE)
    g = torch.Generator(device=DEVICE)
    out = {}
    for K in Ks:
        eps_list = [0.0] if K == 1 else epss
        for eps in eps_list:
            g.manual_seed(7)
            noise = torch.zeros(1, B, d, device=DEVICE) if K == 1 \
                else eps * torch.randn(K, B, d, generator=g, device=DEVICE)
            cand = (cand0.unsqueeze(0) + noise).reshape(K * B, d)
            Cr = Ct.unsqueeze(0).expand(K, B, -1).reshape(K * B, -1)
            Wr = Wt.unsqueeze(0).expand(K, B, -1).reshape(K * B, -1)
            _, ipc = nearest_ball(Cr, cand, n_ball, d)
            xf = radial_bisect(Cr, ipc, cand, n_ball, d)
            viol = ineq_resid(Cr, xf, n_ball, d).reshape(K, B)
            val = obj_fn(Wr, xf).reshape(K, B)
            ballidx, _ = nearest_ball(Cr, xf, n_ball, d); ballidx = ballidx.reshape(K, B)
            merit = val + 1e6 * viol
            bi = merit.argmin(dim=0)
            out[(K, eps)] = (float((val[bi, ar] - vopt).mean()),
                             float((ballidx[bi, ar] != bb).double().mean() * 100))
    return out


def main():
    print("=== Disconnected-ball: does the multimodal gain survive higher dim? ===", flush=True)
    print(f"device: {DEVICE}  (n_ball=4, converged w/ cosine LR decay)\n", flush=True)
    n_ball = 4
    iters_for = {2: 240000, 4: 240000, 8: 300000, 16: 360000}  # more dims -> a bit more training
    print(f"{'d':>3} {'iters':>7} | {'K=1 optgap':>11} {'K=1 wrong%':>11} | {'best K=100 optgap':>18} {'(eps)':>6} {'K=100 wrong%':>13} | {'gain(K1-K100)':>14}", flush=True)
    for d in [2, 4, 8, 16]:
        iters = iters_for[d]
        rng_t = np.random.default_rng(999 + d)
        Ct, Wt = sample(4096, n_ball, d, rng_t)
        vopt, bb = brute_force_opt(Ct, Wt, n_ball, d)
        t0 = time.time()
        m = train(n_ball, d, iters, 128, 4, seed=42)
        r = eval_dim(m, n_ball, d, Ct, Wt, vopt, bb)
        k1 = r[(1, 0.0)]
        # best K=100 over eps by optgap
        best_eps, best = min(((e, r[(100, e)]) for e in (0.1, 0.3, 0.5, 1.0)), key=lambda kv: kv[1][0])
        gain = k1[0] - best[0]
        print(f"{d:>3} {iters:>7} | {k1[0]:11.4f} {k1[1]:10.1f}% | {best[0]:18.4f} {best_eps:>6.2f} {best[1]:12.1f}% | {gain:14.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print("\nReading: if 'gain' collapses toward 0 as d grows -> 2D result was random-search-inflated.", flush=True)
    print("If K=1 wrong% stays high and gain persists -> multimodal routing effect is robust to dim.", flush=True)
    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
