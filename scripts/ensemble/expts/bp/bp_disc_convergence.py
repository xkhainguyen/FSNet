"""Definitive convergence test for the disconnected-ball routing-error question.

Train the BEST architecture (128x4, smallest -> trains fastest, lowest wrong%)
to ACTUAL convergence with LR decay, out to 480k iters, and watch whether K=1
wrong-ball% floors at a positive value (-> structural / irreducible) or keeps
heading toward zero (-> it was undertraining, like HardNet).

This is the test that settles whether the disconnected-ball perturbation gain is
a genuine convergence-surviving multimodal effect or another undertraining
artifact.
"""
from __future__ import annotations
import os, sys, pathlib, time
import importlib.util
import numpy as np
import torch

ROOT = pathlib.Path("/orcd/scratch/orcd/008/khain/FSNet/third-party/Bisection-Projection")
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

# reuse machinery from the controls module
_spec = importlib.util.spec_from_file_location(
    "ctrl", "/orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/bp/bp_disc_controls.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)

torch.set_default_dtype(torch.float64)
DEVICE = C.DEVICE


def train_decay(data, iters, h, nl, seed, lr0=1e-3):
    """Train with cosine LR decay so 'converged' is meaningful."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = C.Solver(data.c_dim + 2, h, nl).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr0 * 0.01)
    for it in range(iters):
        c, w = C.sample_instances(data, 256, rng)
        x = model(c, w)
        lam = 1.0 + 99.0 * min(1.0, it / (0.3 * iters))  # ramp penalty over first 30%
        loss = (C.obj_fn(w, x) + lam * C.feasible(data, c, x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def main():
    print(f"=== Disconnected-ball DEFINITIVE convergence test (128x4 + cosine LR decay) ===", flush=True)
    print(f"device: {DEVICE}", flush=True)
    print("Q: does K=1 wrong-ball% FLOOR (structural) or -> 0 (undertraining)?\n", flush=True)
    data = C.make_data(4); data.to_device(DEVICE)
    Ct, Wt = C.sample_instances(data, 4096, np.random.default_rng(12345))
    _, vopt, bb = C.brute_force_opt(data, Ct, Wt)
    print(f"exact optimum mean wᵀx* = {float(vopt.mean()):.4f}", flush=True)
    print(f"{'iters':>8} | {'K=1 optgap':>11} {'K=1 wrong%':>11} | {'K=100e0.1 optgap':>17} | {'K=100e1 optgap':>15} {'K=100e1 wrong%':>15}", flush=True)
    for iters in [60000, 120000, 240000, 480000]:
        t0 = time.time()
        m = train_decay(data, iters, 128, 4, seed=42)
        r = C.eval_sweep(data, m, Ct, Wt, vopt, bb, Ks=(1, 100), epss=(0.0, 0.1, 1.0))
        k1 = r[(1, 0.0)]; k100e01 = r[(100, 0.1)]; k100e1 = r[(100, 1.0)]
        print(f"{iters:>8} | {k1[0]:11.4f} {k1[1]:10.1f}% | {k100e01[0]:17.4f} | {k100e1[0]:15.4f} {k100e1[1]:14.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
