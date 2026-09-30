"""Did 10 seeds converge to 10 different minima? M1, M2, M4 (L1 penalty), held-out.
  weight distance    ||theta_a - theta_b|| / ||theta_a||  (permutation copies can be far apart)
  function distance  mean |y_a(x) - y_b(x)| over held-out x, vs the models' own label error
                     mean |y_a(x) - y*(x)|; if seeds disagree about as much as they miss
                     the labels, they are genuinely different solutions, not neuron permutations.
"""
import glob
import os

import numpy as np
import torch

from compute_landscape_compare import load_problem, load_model

import argparse
from fig1_common import ckpt, SPEC
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["M1", "M2", "M4"])
ap.add_argument("--nseeds", type=int, default=10)
ap.add_argument("--split", choices=["train", "test"], default="test")
args = ap.parse_args()
prob = load_problem()
X, Y = [t[:1000].to(prob.device) for t in (prob.test_dataset if args.split == "test" else prob.train_dataset).tensors]
A = prob.A
_, _, Vh = torch.linalg.svd(A)
Pn = Vh[50:].T @ Vh[50:]
print("| model | seeds | weight distance (median) | function distance between seeds (median) | "
      "range-space part | null-space part | label error of a seed |")
print("|---|---:|---:|---:|---:|---:|---:|")
for m in args.models:
    th, ys = [], []
    for s in range(args.nseeds):
        net, _ = load_model(prob, ckpt(m, s))
        th.append(torch.nn.utils.parameters_to_vector(net.parameters()).detach())
        with torch.no_grad():
            ys.append(prob.scale(net(X)))
    wd, fd, fr, fnull = [], [], [], []
    for a in range(len(th)):
        for b in range(a + 1, len(th)):
            wd.append(((th[a] - th[b]).norm() / th[a].norm()).item())
            e = ys[a] - ys[b]
            fd.append(e.abs().mean().item())
            fnull.append((e @ Pn).norm(dim=1).mean().item())
            fr.append((e - e @ Pn).norm(dim=1).mean().item())
    lab = np.mean([(y - Y).abs().mean().item() for y in ys])
    print(f"| {m} | {len(th)} | {np.median(wd):.3f} | {np.median(fd):.3f} | {np.median(fr):.3f} | {np.median(fnull):.3f} | {lab:.3f} |")
