"""Score trained runs on fit quality: label error, objective, violation, merit.

  python screen_landscape_fit.py --glob '*_MLP_sup_pen_seed0_nepochs3000_*dropout*'
"""
import argparse
import glob
import os

import torch

from compute_landscape_compare import load_problem, load_model
from utils.lbfgs import nondiff_lbfgs_solve

D = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"


def score(prob, model, X, Y, fs):
    with torch.no_grad():
        y = prob.scale(model(X))
    out = {}
    for tag, yy in [("raw", y)] + ([("fs", nondiff_lbfgs_solve(X, y, prob, val_tol=1e-9, memory=30,
                                                                 max_iter=50, scale=1000).detach())] if fs else []):
        with torch.no_grad():
            obj, obj_star = prob.obj_fn(yy), prob.obj_fn(Y)
            viol = prob.eq_resid(X, yy).abs().sum(1) + prob.ineq_resid(X, yy).abs().sum(1)
            out[f"err_{tag}"] = (yy - Y).abs().mean().item()
            out[f"gap_{tag}"] = ((obj - obj_star) / obj_star.abs()).mean().item()
            out[f"viol_{tag}"] = viol.mean().item()
            out[f"merit_{tag}"] = (obj + 1e5 * viol).mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True, help="run-dir glob under the SOCP results dir")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--no_fs", action="store_true")
    args = ap.parse_args()

    prob = load_problem()
    Xt, Yt = [t[:args.n_train].to(prob.device) for t in prob.train_dataset.tensors]
    Xv, Yv = [t.to(prob.device) for t in prob.val_dataset.tensors]
    runs = sorted({r for g in args.glob for r in glob.glob(os.path.join(D, g))})

    cols = ["err_raw", "gap_raw", "viol_raw", "merit_raw"] + ([] if args.no_fs else ["err_fs", "merit_fs"])
    print("| run | train err | " + " | ".join(f"val {c}" for c in cols) + " |")
    print("|---|---:|" + "---:|" * len(cols))
    for r in runs:
        path = os.path.join(r, "model.pt")
        if not os.path.exists(path):
            continue
        model, _ = load_model(prob, path)
        tr = score(prob, model, Xt, Yt, fs=False)
        va = score(prob, model, Xv, Yv, fs=not args.no_fs)
        name = os.path.basename(r).split("_MLP_")[1]
        print(f"| {name} | {tr['err_raw']:.3f} | " + " | ".join(f"{va[c]:.4g}" for c in cols) + " |", flush=True)


if __name__ == "__main__":
    main()
