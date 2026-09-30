"""Deployment-time solution quality (held-out): M3 variants are scored after FS (500 iters, tol 1e-9),
M1/M2/M4 on their raw output. merit = obj + 1e5 * L1 violation; opt gap vs the labels' objective."""
import glob, os
import numpy as np
from compute_landscape_compare import load_problem, load_model, eval_components
from fig1_common import ckpt
D = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
prob = load_problem()
X, Y = [t[:500].to(prob.device) for t in prob.test_dataset.tensors]
obj_star = prob.obj_fn(Y).mean().item()
fs = dict(val_tol=1e-9, memory=30, max_iter=500, scale=1000)
print(f"labels: objective {obj_star:.3f}\n")
print("| model | deployed output | seeds | objective | violation (L1) | merit (median) | merit per seed |")
print("|---|---|---:|---:|---:|---:|---|")
rows = []
for m in ["M1", "M2", "M4"]:
    r = []
    for s in range(3):
        net, _ = load_model(prob, ckpt(m, s))
        c = eval_components(net, prob, X, Y, None, 500)
        r.append((c["obj"], c["viol_l1"], c["obj"] + 1e5 * c["viol_l1"]))
    rows.append((m, "raw", r))
for name, tag in [("M3 (rho 10)", "eq10.0_ineq10.0_penl1"), ("M3a (rho 0)", "eq0.0_ineq0.0_penl1"),
                  ("M3c (rho 0 + dist)", "eq0.0_ineq0.0_dist5.0_penl1"), ("M3d (warm-up + dist)", "eq10.0_ineq10.0_warm300_dist5.0_penl1")]:
    r = []
    for s in range(3):
        runs = [x for x in sorted(glob.glob(f"{D}/*_MLP_sup_pen_fs_seed{s}_nepochs3000_lr0.0003_trainsize7000_obj0.1_{tag}_dropout0.0_*")) if os.path.exists(x + "/model.pt")]
        if not runs:
            continue
        net, _ = load_model(prob, runs[-1] + "/model.pt")
        c = eval_components(net, prob, X, Y, fs, 500)
        r.append((c["obj_fs"], c["viol_l1_fs"], c["obj_fs"] + 1e5 * c["viol_l1_fs"]))
    rows.append((name, "after FS (500 it)", r))
for name, dep, r in rows:
    a = np.array(r)
    print(f"| {name} | {dep} | {len(a)} | {np.median(a[:, 0]):.3f} | {np.median(a[:, 1]):.2g} | {np.median(a[:, 2]):.4g} | "
          + ", ".join(f"{v:.3g}" for v in a[:, 2]) + " |", flush=True)
