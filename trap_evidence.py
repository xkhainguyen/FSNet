"""Is it easy to get trapped? (M1 vs M2 vs M4, L1 penalty, held-out)
1. Escape barrier of every local minimum on the wide seed plane: the lowest-barrier path
   (8-connected grid) to any strictly lower minimum, in decades above the minimum itself.
2. Across 10 training seeds: own held-out training loss and raw merit of each converged model.
"""
import glob
import os

import numpy as np
import torch

from compute_landscape_compare import load_problem, load_model
from make_manyminima import own, local_minima
from make_m1m2_barrier import minimax_path

OUT = "figures/landscape/fig1"
D = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
SPEC = {"M1": ("sup_pen", "obj0.1_eq10.0_ineq10.0", "0.0003", 10.0),
        "M2": ("sup_pen", "obj0.1_eq100000.0_ineq100000.0", "0.0001", 1e5),
        "M4": ("penalty", "obj1.0_eq10.0_ineq10.0", "0.001", 10.0)}

print("## 1. Escape barriers on the wide seed plane\n")
print("| model | minimum (decades above best seed) | escape barrier (decades above the minimum) | is a seed |")
print("|---|---:|---:|---|")
for m in SPEC:
    d = dict(np.load(f"{OUT}/seedplane_wide_test_{m}.npz", allow_pickle=True))
    xs, ys, pts = d["xs"], d["ys"], d["points"]
    Z = own(m, d)
    if Z.min() <= 0:
        Z = Z - Z.min() + 1.0
    seeds = {(int(np.argmin(np.abs(ys - p[1]))), int(np.argmin(np.abs(xs - p[0])))) for p in pts}
    zbest = min(Z[k] for k in seeds)
    jm, im = local_minima(Z)
    mins = sorted(zip(jm, im), key=lambda k: Z[k])
    for k, a in enumerate(mins):
        if k == 0:
            print(f"| {m} | {np.log10(Z[a] / zbest):.2f} | (global in slice) | {a in seeds} |")
            continue
        esc = min(max(Z[p] for p in minimax_path(Z, a, b)) for b in mins[:k])
        print(f"| {m} | {np.log10(Z[a] / zbest):.2f} | {np.log10(esc / Z[a]):.2f} | {a in seeds} |")

print("\n## 2. Converged quality across 10 seeds (held-out, 1000 samples)\n")
prob = load_problem()
X, Y = [t[:1000].to(prob.device) for t in prob.test_dataset.tensors]
print("| model | seeds | own loss: min / median / max | max / min | raw merit: min / median / max |")
print("|---|---:|---|---:|---|")
for m, (meth, w, lr, rho) in SPEC.items():
    L, M = [], []
    for s in range(10):
        runs = [r for r in sorted(glob.glob(f"{D}/*_MLP_{meth}_seed{s}_nepochs3000_lr{lr}_trainsize7000_{w}_penl1_dropout0.0_lrschedcosine_etamin1e-06"))
                if os.path.exists(r + "/model.pt")]
        if not runs:
            continue
        net, _ = load_model(prob, runs[-1] + "/model.pt")
        with torch.no_grad():
            y = prob.scale(net(X))
            v = prob.eq_resid(X, y).abs().sum(1) + prob.ineq_resid(X, y).abs().sum(1)
            first = 100 * (lambda e: torch.where(e.abs() <= 0.1, 0.5 * e ** 2 / 0.1, e.abs() - 0.05))(y - Y).mean(1) \
                if meth == "sup_pen" else prob.obj_fn(y)
            L.append((first + rho * v).mean().item())
            M.append((prob.obj_fn(y) + 1e5 * v).mean().item())
    L, M = np.array(L), np.array(M)
    print(f"| {m} | {len(L)} | {L.min():.4g} / {np.median(L):.4g} / {L.max():.4g} | {L.max() / L.min():.2f} | "
          f"{M.min():.3g} / {np.median(M):.3g} / {M.max():.3g} |")
