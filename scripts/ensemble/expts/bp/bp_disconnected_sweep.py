"""Does multistart give a CONVERGENCE-SURVIVING gain on a multimodal feasible set?

Testbed: amortized linear optimization over a UNION OF 4 DISJOINT BALLS (BP's
`Disconnected_Ball`). The feasible set is genuinely disconnected, so the optimal
ball is a *discontinuous* function of (geometry c, objective direction w) — the
multimodality that FSNet/DC3/Πnet/HardNet (all convex feasible sets) never had.

Problem per instance:
    min  wᵀx   s.t.  x ∈ ⋃_{i=1..4} Ball(center_i, radius_i)
Conditioning input to the solver: [c (4 centers+radii = 12d), w (2d)] -> x (2d).

Repair: radial bisection (BP `ip_bisection`) from the nearest feasible ball-center
interior point toward the candidate. Under K perturbations, perturbations near a
ball boundary pick different nearest-ball IPs -> land in different balls ->
multistart explores modes; best_merit (lowest feasible wᵀx) picks.

We train to convergence and ask whether the K=100 perturbation gain SURVIVES.
If large -> ensembling genuinely helps on multimodal problems (positive regime).
If ~1% -> the negative result generalizes even here.
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
    """max constraint violation per sample (0 = feasible). x:(B,2) c:(B,12)"""
    return data.ineq_resid(c, x).view(x.shape[0])  # already clamped min-over-balls


def obj_fn(w, x):
    return (w * x).sum(dim=1)


def nearest_ball_idx(c, x, n_ball=4):
    """Index of ball whose center is nearest to x. c:(B,12) x:(B,2) -> (B,)"""
    B = c.shape[0]
    centers = c[:, : n_ball * 2].view(B, n_ball, 2)
    d = torch.norm(x.view(B, 1, 2) - centers, dim=-1)
    return d.argmin(dim=1)


def nearest_ball_center(c, x, n_ball=4):
    """Center of the ball whose center is nearest to x. c:(B,12) x:(B,2)"""
    B = c.shape[0]
    centers = c[:, : n_ball * 2].view(B, n_ball, 2)
    idx = nearest_ball_idx(c, x, n_ball)
    return centers[torch.arange(B, device=c.device), idx]


def radial_bisect(data, c, ip, cand, n_steps=60, bis=0.9, eps=1e-7):
    """Largest alpha in [0,1] s.t. ip + alpha*(cand-ip) feasible. Returns boundary pt.
    Bisection converges geometrically; n_steps=60 -> alpha gap < 1e-27."""
    B = c.shape[0]
    lo = torch.zeros(B, 1, device=c.device)
    hi = torch.ones(B, 1, device=c.device)
    for _ in range(n_steps):
        a = (1 - bis) * lo + bis * hi
        xt = a * (cand - ip) + ip
        feas = (feasible(data, c, xt) < eps).view(B, 1)
        lo = torch.where(feas, a, lo)
        hi = torch.where(~feas, a, hi)
        if (hi - lo).max() < 1e-12:
            break
    return lo * (cand - ip) + ip


class Solver(nn.Module):
    def __init__(self, cin=14, h=128, nl=4):
        super().__init__()
        layers = [nn.Linear(cin, h), nn.ReLU()]
        for _ in range(nl - 1):
            layers += [nn.Linear(h, h), nn.ReLU()]
        layers += [nn.Linear(h, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, c, w):
        return self.net(torch.cat([c, w], dim=1))


def sample_instances(data, n, rng):
    """c ~ data.sampling_range, w ~ unit circle."""
    lo, hi = data.sampling_range[0], data.sampling_range[1]
    c = rng.random((n, data.c_dim)) * (hi - lo) + lo
    theta = rng.random(n) * 2 * np.pi
    w = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return torch.tensor(c, device=DEVICE), torch.tensor(w, device=DEVICE)


def brute_force_opt(data, c, w, n_ball=4):
    """Exact optimum: for linear obj over union of balls, optimum of each ball is
    center_i - radius_i * w (w unit). Pick the ball giving lowest wᵀx."""
    B = c.shape[0]
    centers = c[:, : n_ball * 2].view(B, n_ball, 2)
    radii = c[:, n_ball * 2 :].view(B, n_ball)
    wn = w / w.norm(dim=1, keepdim=True)
    # candidate optimum on each ball boundary: center_i - r_i * w
    cand = centers - radii.unsqueeze(-1) * wn.unsqueeze(1)  # (B, n_ball, 2)
    vals = (cand * wn.unsqueeze(1)).sum(dim=-1)  # (B, n_ball)
    best = vals.argmin(dim=1)
    xopt = cand[torch.arange(B, device=c.device), best]
    return xopt, obj_fn(w, xopt), best


def main(iters: int, seed: int):
    print(f"=== BP disconnected-ball multimodal sweep: seed={seed}, iters={iters} ===", flush=True)
    print(f"device: {DEVICE}", flush=True)
    data = Disconnected_Ball(n_ball=4)
    data.to_device(DEVICE)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Fixed test set
    Ctest, Wtest = sample_instances(data, 2048, np.random.default_rng(seed + 999))
    xopt, vopt, best_ball = brute_force_opt(data, Ctest, Wtest)
    print(f"exact optimum: mean wᵀx* = {float(vopt.mean()):.4f}", flush=True)

    model = Solver().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 256

    print(f"Training {iters} iters (penalty ramped) ...", flush=True)
    t0 = time.time()
    for it in range(iters):
        c, w = sample_instances(data, bs, rng)
        x = model(c, w)
        lam = 1.0 + 99.0 * min(1.0, it / (0.5 * iters))  # ramp 1 -> 100
        viol = feasible(data, c, x)
        loss = (obj_fn(w, x) + lam * viol ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % max(1, iters // 10) == 0 or it == 0:
            with torch.no_grad():
                xv = model(Ctest, Wtest)
                vfeas = feasible(data, Ctest, xv)
                gap = (obj_fn(Wtest, xv) - vopt)
            print(f"  it {it+1:5d}/{iters} loss={float(loss):.4f} lam={lam:.0f} | "
                  f"test viol_mean={float(vfeas.mean()):.3e} raw_optgap={float(gap.mean()):.4f}", flush=True)
    print(f"trained in {time.time()-t0:.1f}s", flush=True)

    # ---- Eval: K-perturbation sweep, repair = radial bisection ----
    model.eval()
    with torch.no_grad():
        cand0 = model(Ctest, Wtest)  # raw NN candidate (may be infeasible)

    def repaired_metrics(K, eps, rng_t):
        B = Ctest.shape[0]
        if K == 1:
            noise = torch.zeros(1, B, 2, device=DEVICE)
        else:
            noise = eps * torch.randn(K, B, 2, generator=rng_t, device=DEVICE)
        cand = cand0.unsqueeze(0) + noise               # (K,B,2)
        cflat = cand.reshape(K * B, 2)
        Crep = Ctest.unsqueeze(0).expand(K, B, -1).reshape(K * B, -1)
        Wrep = Wtest.unsqueeze(0).expand(K, B, -1).reshape(K * B, -1)
        ip = nearest_ball_center(Crep, cflat)           # per-candidate nearest ball
        xfeas = radial_bisect(data, Crep, ip, cflat)    # feasible boundary point
        viol = feasible(data, Crep, xfeas).reshape(K, B)
        val = obj_fn(Wrep, xfeas).reshape(K, B)
        ball = nearest_ball_idx(Crep, xfeas).reshape(K, B)  # which ball each landed in
        merit = val + 1e6 * viol
        bi = merit.argmin(dim=0)
        ar = torch.arange(B, device=DEVICE)
        return val[bi, ar], viol[bi, ar], ball[bi, ar]

    g = torch.Generator(device=DEVICE)
    print(f"\n--- K-perturbation sweep (best_merit; repair=radial bisection, n_steps=60) ---", flush=True)
    print(f"exact optimum mean = {float(vopt.mean()):.4f}", flush=True)
    # wrong-ball% = solution landed in a ball != the objective-optimal ball (ROUTING error,
    #   a genuine multimodal failure that should survive convergence).
    # optgap = mean objective gap to exact optimum (mixes routing + precision error).
    print(f"{'K':>4} {'eps':>6} | {'mean wᵀx':>10} {'viol':>10} {'optgap':>10} {'wrong-ball%':>12}", flush=True)
    for K in [1, 5, 20, 100]:
        for eps in [0.0, 0.05, 0.1, 0.3, 0.5, 1.0]:
            if K == 1 and eps != 0.0:
                continue
            g.manual_seed(7)
            val, viol, ball = repaired_metrics(K, eps, g)
            optgap = (val - vopt)
            wrongball = (ball != best_ball).double().mean() * 100
            print(f"{K:>4} {eps:>6.2f} | {float(val.mean()):10.4f} {float(viol.mean()):10.3e} "
                  f"{float(optgap.mean()):10.4f} {float(wrongball):11.1f}%", flush=True)
    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.iters, a.seed)
