"""Pick the Fig. 1 checkpoints: per model, the lr with the lowest own training loss.

Models (all L1 penalty, seed 0, 3000 epochs, dropout 0):
  M1 SL, small rho (sup_pen, 10) | M2 SL, large rho (sup_pen, 1e5)
  M3 SL + FS, small rho (sup_pen_fs, 10) | M4 SSL, small rho (penalty, 10)
Own loss = Trainer.compute_batch_loss with the checkpoint's config (exactly what was
optimized), averaged over n samples. Writes figures/landscape/fig1/ckpts.json.
"""
import glob
import json
import os

import torch

from compute_landscape_compare import load_problem, load_model
from utils.lbfgs import nondiff_lbfgs_solve
from utils.trainer import Trainer

D = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
MODELS = {
    "M1": ("sup_pen", "obj0.1_eq10.0_ineq10.0"),
    "M2": ("sup_pen", "obj0.1_eq100000.0_ineq100000.0"),
    "M3": ("sup_pen_fs", "obj0.1_eq10.0_ineq10.0"),
    "M4": ("penalty", "obj1.0_eq10.0_ineq10.0"),
}
LRS = ["0.0001", "0.0003", "0.001"]
N = 2000


def own_loss(prob, model, cfg, X, Y, chunk=500):
    cfg = dict(cfg)
    cfg.setdefault("checkpoint", None)
    tr = Trainer(prob, cfg)
    tot = 0.0
    for s in range(0, len(X), chunk):
        x, y = X[s:s + chunk], Y[s:s + chunk]
        pred = model(x)  # graph kept: the FS solver differentiates through its input
        loss, _ = tr.compute_batch_loss(x, pred, y, {"epoch": 0})
        tot += loss.sum().item()
    return tot / len(X)


def merit_fs(prob, model, X):
    with torch.no_grad():
        y = prob.scale(model(X))
    y = nondiff_lbfgs_solve(X, y, prob, val_tol=1e-9, memory=30, max_iter=50, scale=1000).detach()
    v = prob.eq_resid(X, y).abs().sum(1) + prob.ineq_resid(X, y).abs().sum(1)
    return (prob.obj_fn(y) + 1e5 * v).mean().item()


def main():
    prob = load_problem()
    Xt, Yt = [t[:N].to(prob.device) for t in prob.train_dataset.tensors]
    Xs, Ys = [t[:1000].to(prob.device) for t in prob.test_dataset.tensors]
    chosen = {}
    print("| model | lr | own loss (train) | own loss (test) | merit after FS (test) |")
    print("|---|---|---:|---:|---:|")
    for name, (method, w) in MODELS.items():
        best = None
        for lr in LRS:
            pat = f"{D}/*_MLP_{method}_seed0_nepochs3000_lr{lr}_trainsize7000_{w}_penl1_dropout0.0_lrschedcosine_etamin1e-06"
            runs = [r for r in sorted(glob.glob(pat)) if os.path.exists(r + "/model.pt")]
            if not runs:
                print(f"| {name} | {lr} | missing | | |")
                continue
            model, cfg = load_model(prob, runs[-1] + "/model.pt")
            ltr, lte, mfs = own_loss(prob, model, cfg, Xt, Yt), own_loss(prob, model, cfg, Xs, Ys), merit_fs(prob, model, Xs)
            print(f"| {name} | {lr} | {ltr:.4g} | {lte:.4g} | {mfs:.4g} |", flush=True)
            if best is None or ltr < best[0]:
                best = (ltr, runs[-1] + "/model.pt", lr)
        if best:
            chosen[name] = {"ckpt": best[1], "lr": best[2], "own_train_loss": best[0]}
    os.makedirs("figures/landscape/fig1", exist_ok=True)
    with open("figures/landscape/fig1/ckpts.json", "w") as f:
        json.dump(chosen, f, indent=1)
    print("\nchosen:", {k: v["lr"] for k, v in chosen.items()})


if __name__ == "__main__":
    main()
