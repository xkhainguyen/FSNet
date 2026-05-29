"""Controls A & B for the disconnected-ball positive result.

A — capacity × training scaling: is the K=1 wrong-ball rate STRUCTURAL (intrinsic
    to the discontinuous argmin-ball map) or just under-capacity/undertraining?
    Sweep {net width × depth} × {train iters} and watch K=1 wrong-ball% + optgap.
    Plateau at a positive floor independent of capacity -> structural.
    Keeps dropping with capacity/iters -> it was under-convergence (like HardNet).

B — connected control: same pipeline with a SINGLE ball (convex, connected
    feasible set). The perturbation gain should VANISH -> confirms the effect is
    the disconnectedness, not the pipeline/eval.

Also reports the eps decomposition (ball-switching vs within-ball precision) so
we can see whether the gain at realistic small eps is routing or just precision.
"""
from __future__ import annotations
import os, sys, pathlib, time
import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path("/orcd/scratch/orcd/008/khain/FSNet/third-party/Bisection-Projection")
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from utils.toy_utils import Disconnected_Ball  # noqa: E402

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feasible(data, c, x):
    return data.ineq_resid(c, x).view(x.shape[0])


def obj_fn(w, x):
    return (w * x).sum(dim=1)


def nearest_ball_idx(c, x, n_ball):
    B = c.shape[0]
    centers = c[:, : n_ball * 2].view(B, n_ball, 2)
    d = torch.norm(x.view(B, 1, 2) - centers, dim=-1)
    return d.argmin(dim=1)


def nearest_ball_center(c, x, n_ball):
    B = c.shape[0]
    centers = c[:, : n_ball * 2].view(B, n_ball, 2)
    idx = nearest_ball_idx(c, x, n_ball)
    return centers[torch.arange(B, device=c.device), idx]


def radial_bisect(data, c, ip, cand, n_steps=60, bis=0.9, eps=1e-7):
    B = c.shape[0]
    lo = torch.zeros(B, 1, device=c.device); hi = torch.ones(B, 1, device=c.device)
    for _ in range(n_steps):
        a = (1 - bis) * lo + bis * hi
        xt = a * (cand - ip) + ip
        fe = (feasible(data, c, xt) < eps).view(B, 1)
        lo = torch.where(fe, a, lo); hi = torch.where(~fe, a, hi)
        if (hi - lo).max() < 1e-12:
            break
    return lo * (cand - ip) + ip


class Solver(nn.Module):
    def __init__(self, cin, h, nl):
        super().__init__()
        layers = [nn.Linear(cin, h), nn.ReLU()]
        for _ in range(nl - 1):
            layers += [nn.Linear(h, h), nn.ReLU()]
        layers += [nn.Linear(h, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, c, w):
        return self.net(torch.cat([c, w], dim=1))


def make_data(n_ball):
    """Disconnected_Ball with n_ball components (n_ball=1 -> connected/convex control)."""
    d = Disconnected_Ball(n_ball=n_ball)
    # Disconnected_Ball hardcodes n_ball=4 internally in some attrs; rebuild ranges for n_ball.
    d.n_ball = n_ball
    d.c_dim = n_ball * 3
    d.sampling_range = np.zeros([2, n_ball * 3])
    d.sampling_range[0, : 2 * n_ball] = -1
    d.sampling_range[1, : 2 * n_ball] = 1
    d.sampling_range[0, 2 * n_ball:] = 0.5
    d.sampling_range[1, 2 * n_ball:] = 0.7
    return d


def sample_instances(data, n, rng):
    lo, hi = data.sampling_range[0], data.sampling_range[1]
    c = rng.random((n, data.c_dim)) * (hi - lo) + lo
    theta = rng.random(n) * 2 * np.pi
    w = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return torch.tensor(c, device=DEVICE), torch.tensor(w, device=DEVICE)


def brute_force_opt(data, c, w):
    nb = data.n_ball
    B = c.shape[0]
    centers = c[:, : nb * 2].view(B, nb, 2)
    radii = c[:, nb * 2:].view(B, nb)
    wn = w / w.norm(dim=1, keepdim=True)
    cand = centers - radii.unsqueeze(-1) * wn.unsqueeze(1)
    vals = (cand * wn.unsqueeze(1)).sum(dim=-1)
    best = vals.argmin(dim=1)
    xopt = cand[torch.arange(B, device=c.device), best]
    return xopt, obj_fn(w, xopt), best


def train(data, iters, h, nl, seed):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = Solver(data.c_dim + 2, h, nl).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(iters):
        c, w = sample_instances(data, 256, rng)
        x = model(c, w)
        lam = 1.0 + 99.0 * min(1.0, it / (0.5 * iters))
        loss = (obj_fn(w, x) + lam * feasible(data, c, x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_sweep(data, model, Ctest, Wtest, vopt, best_ball, Ks=(1, 100), epss=(0.0, 0.1, 0.5, 1.0)):
    model.eval()
    with torch.no_grad():
        cand0 = model(Ctest, Wtest)
    g = torch.Generator(device=DEVICE)
    rows = {}
    for K in Ks:
        for eps in epss:
            if K == 1 and eps != 0.0:
                continue
            g.manual_seed(7)
            B = Ctest.shape[0]
            noise = torch.zeros(1, B, 2, device=DEVICE) if K == 1 \
                else eps * torch.randn(K, B, 2, generator=g, device=DEVICE)
            cand = cand0.unsqueeze(0) + noise
            cflat = cand.reshape(K * B, 2)
            Crep = Ctest.unsqueeze(0).expand(K, B, -1).reshape(K * B, -1)
            Wrep = Wtest.unsqueeze(0).expand(K, B, -1).reshape(K * B, -1)
            ip = nearest_ball_center(Crep, cflat, data.n_ball)
            xf = radial_bisect(data, Crep, ip, cflat)
            viol = feasible(data, Crep, xf).reshape(K, B)
            val = obj_fn(Wrep, xf).reshape(K, B)
            ball = nearest_ball_idx(Crep, xf, data.n_ball).reshape(K, B)
            merit = val + 1e6 * viol
            bi = merit.argmin(dim=0); ar = torch.arange(B, device=DEVICE)
            optgap = float((val[bi, ar] - vopt).mean())
            wrong = float((ball[bi, ar] != best_ball).double().mean() * 100)
            rows[(K, eps)] = (optgap, wrong)
    return rows


def main():
    print(f"=== Disconnected-ball CONTROLS A (capacity×iters) + B (connected) ===", flush=True)
    print(f"device: {DEVICE}", flush=True)

    # ---------- Control A: capacity × training-iters scaling, n_ball=4 ----------
    print("\n##### CONTROL A: is K=1 wrong-ball% structural or under-capacity/undertraining? #####", flush=True)
    print("(if it keeps dropping as net/iters grow -> under-convergence; if it floors -> structural)\n", flush=True)
    data4 = make_data(4); data4.to_device(DEVICE)
    Ct, Wt = sample_instances(data4, 2048, np.random.default_rng(12345))
    xopt, vopt, bb = brute_force_opt(data4, Ct, Wt)
    print(f"{'net (h x nl)':>14} {'iters':>7} | {'K=1 optgap':>11} {'K=1 wrong%':>11} | {'K=100e1 optgap':>15} {'K=100e1 wrong%':>15}", flush=True)
    configs = [
        (128, 4, 60000),    # baseline (matches prior run)
        (128, 4, 120000),   # 2x iters
        (256, 6, 60000),    # bigger net
        (256, 6, 120000),   # bigger net + 2x iters
        (512, 6, 120000),   # biggest
    ]
    for (h, nl, iters) in configs:
        t0 = time.time()
        m = train(data4, iters, h, nl, seed=42)
        r = eval_sweep(data4, m, Ct, Wt, vopt, bb)
        k1 = r[(1, 0.0)]; k100 = r[(100, 1.0)]
        print(f"{h:>6}x{nl:<7} {iters:>7} | {k1[0]:11.4f} {k1[1]:10.1f}% | {k100[0]:15.4f} {k100[1]:14.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    # ---------- Control B: connected single ball (convex) ----------
    print("\n##### CONTROL B: connected single ball (convex) — gain should VANISH #####\n", flush=True)
    data1 = make_data(1); data1.to_device(DEVICE)
    Ct1, Wt1 = sample_instances(data1, 2048, np.random.default_rng(54321))
    _, vopt1, bb1 = brute_force_opt(data1, Ct1, Wt1)
    m1 = train(data1, 60000, 128, 4, seed=42)
    r1 = eval_sweep(data1, m1, Ct1, Wt1, vopt1, bb1, epss=(0.0, 0.1, 0.5, 1.0))
    print(f"{'K':>4} {'eps':>5} | {'optgap':>10} {'wrong-ball%':>12}", flush=True)
    for (K, eps), (og, wr) in r1.items():
        print(f"{K:>4} {eps:>5.2f} | {og:10.4f} {wr:11.1f}%", flush=True)

    # ---------- eps decomposition on the 4-ball baseline (routing vs precision) ----------
    print("\n##### eps decomposition (4-ball, 128x4, 120k): ball-switching vs within-ball precision #####\n", flush=True)
    m4 = train(data4, 120000, 128, 4, seed=42)
    rdec = eval_sweep(data4, m4, Ct, Wt, vopt, bb, Ks=(1, 100), epss=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0))
    print(f"{'K':>4} {'eps':>5} | {'optgap':>10} {'wrong-ball%':>12}", flush=True)
    for (K, eps) in sorted(rdec.keys()):
        og, wr = rdec[(K, eps)]
        print(f"{K:>4} {eps:>5.2f} | {og:10.4f} {wr:11.1f}%", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
