"""Decisive per-dimension convergence check at d=4.

The high-dim sweep gave d=4 the same ~240k iters as d=2, so its K=1 wrong-ball
(61%) is confounded by undertraining (K=1 optgap 0.46 = far from converged).
Question: does d=4 K=1 wrong-ball PLATEAU high (~structural, dimension amplifies
multimodal difficulty) or drop toward the d=2 level ~6% (undertraining)?

Train d=4 to real convergence: 240k -> 480k -> 960k iters, cosine LR decay.
"""
from __future__ import annotations
import time, importlib.util
import numpy as np
import torch

_spec = importlib.util.spec_from_file_location(
    "hd", "/orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/bp/bp_disc_highdim.py")
H = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(H)

DEVICE = H.DEVICE


def main():
    print("=== d=4 convergence check (does K=1 wrong-ball plateau or drop to ~6%?) ===", flush=True)
    print(f"device: {DEVICE}  n_ball=4, d=4, cosine LR decay\n", flush=True)
    n_ball, d = 4, 4
    rng_t = np.random.default_rng(999 + d)
    Ct, Wt = H.sample(4096, n_ball, d, rng_t)
    vopt, bb = H.brute_force_opt(Ct, Wt, n_ball, d)
    print(f"{'iters':>8} | {'K=1 optgap':>11} {'K=1 wrong%':>11} | {'best K=100 optgap':>18} {'K=100 wrong%':>13}", flush=True)
    for iters in [240000, 480000, 960000]:
        t0 = time.time()
        m = H.train(n_ball, d, iters, 128, 4, seed=42)
        r = H.eval_dim(m, n_ball, d, Ct, Wt, vopt, bb)
        k1 = r[(1, 0.0)]
        be, best = min(((e, r[(100, e)]) for e in (0.1, 0.3, 0.5, 1.0)), key=lambda kv: kv[1][0])
        print(f"{iters:>8} | {k1[0]:11.4f} {k1[1]:10.1f}% | {best[0]:18.4f} {best[1]:12.1f}%  (eps={be:.2f}, {time.time()-t0:.0f}s)", flush=True)
    print("\nIf K=1 wrong% -> ~6% (d=2 level): the 61% was undertraining; dimension does NOT add a structural floor.", flush=True)
    print("If it plateaus high: dimension genuinely amplifies multimodal routing difficulty.", flush=True)
    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
