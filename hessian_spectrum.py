"""Scale-invariant conditioning / smoothness of each model's training loss.

At the trained weights theta*, with Hessian-vector products from central
finite differences of the training gradient (so it works through the FS layer):
  lam_max      top Hessian eigenvalue (power iteration)
  lam_min      bottom eigenvalue (power iteration on H - lam_max I)
  c_rand       mean curvature v'Hv along unit Gaussian directions
  stiffness    lam_max / c_rand: stiff-vs-typical anisotropy, invariant to loss scale
  nonconvex    lam_min / lam_max (negative = saddle-like, Li et al. ratio)
  kink         c_rand(eps small) / c_rand(eps large): ~1 for smooth losses, grows
               as eps shrinks when the loss has kinks (L1 penalties, FS switching)
The loss is each model's own training loss (Trainer.compute_batch_loss with
the checkpoint's config) on a fixed subset of the training set; the merit
obj + 1e5 * L1 violation is also measured at the same weights.

  python hessian_spectrum.py --ckpt sl_small=path/model.pt --out figures/landscape/hessian_sl_small.json
"""
import argparse
import json

import numpy as np
import torch

from compute_landscape_compare import load_problem, load_model
from utils.trainer import Trainer


def make_losses(prob, model, cfg, X, Y):
    cfg = dict(cfg)
    cfg.setdefault("checkpoint", None)
    trainer = Trainer(prob, cfg)
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    sizes = [p.numel() for _, p in model.named_parameters()]

    def forward(theta):
        chunks = torch.split(theta, sizes)
        return torch.func.functional_call(model, {n: c.view(s) for n, c, s in zip(names, chunks, shapes)}, (X,))

    def own(theta):
        loss, _ = trainer.compute_batch_loss(X, forward(theta), Y, {"epoch": 0})
        return loss.mean()

    def merit(theta):
        y = prob.scale(forward(theta))
        if cfg["method"] == "sup_pen_fs":
            from utils.lbfgs import hybrid_lbfgs_solve
            c = cfg[cfg["method"]]
            y = hybrid_lbfgs_solve(X, y, prob, val_tol=c["val_tol"], memory=c["memory_size"],
                                   max_iter=c["max_iter"], max_diff_iter=c["max_diff_iter"], scale=c["scale"])
        viol = prob.eq_resid(X, y).abs().sum(1) + prob.ineq_resid(X, y).abs().sum(1)
        return (prob.obj_fn(y) + 1e5 * viol).mean()

    return own, merit


def grad(f, theta):
    theta = theta.detach().requires_grad_(True)
    with torch.enable_grad():
        g, = torch.autograd.grad(f(theta), theta)
    return g.detach()


def hvp(f, theta, v, h):
    return (grad(f, theta + h * v) - grad(f, theta - h * v)) / (2 * h)


def power(f, theta, h, iters, shift=0.0, seed=0, return_vec=False):
    g = torch.Generator(device=theta.device).manual_seed(seed)
    v = torch.randn(theta.shape, generator=g, device=theta.device, dtype=theta.dtype)
    v /= v.norm()
    lam = 0.0
    for _ in range(iters):
        w = hvp(f, theta, v, h) - shift * v
        lam = (v @ w).item()
        v = w / (w.norm() + 1e-30)
    return (lam + shift, v) if return_vec else lam + shift


def rand_curv(f, theta, h, n, seed=1):
    g = torch.Generator(device=theta.device).manual_seed(seed)
    out = []
    for _ in range(n):
        v = torch.randn(theta.shape, generator=g, device=theta.device, dtype=theta.dtype)
        v /= v.norm()
        out.append((v @ hvp(f, theta, v, h)).item())
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="name=path/to/model.pt")
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--save_vec", default=None, help="save the own-loss top Hessian eigenvector (.pt)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--n_rand", type=int, default=5)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4],
                    help="FD step as a fraction of ||theta*||; first is the reference")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    name, path = args.ckpt.split("=", 1)
    prob = load_problem()
    model, cfg = load_model(prob, path)
    ds = prob.train_dataset if args.split == "train" else prob.test_dataset
    X, Y = [t[:args.n_eval].to(prob.device) for t in ds.tensors]
    theta = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    tn = theta.norm().item()
    own, merit = make_losses(prob, model, cfg, X, Y)

    res = {"name": name, "path": path, "theta_norm": tn, "split": args.split}
    for tag, f in [("own", own), ("merit", merit)]:
        res[f"{tag}_value"] = f(theta.clone().requires_grad_(True)).item()  # FS solver needs a graph through y
        h = args.eps[0] * tn
        lam_max, vec = power(f, theta, h, args.iters, return_vec=True)
        if tag == "own" and args.save_vec:
            torch.save(vec.cpu(), args.save_vec)  # top eigenvector of the own loss, unit norm
        lam_min = power(f, theta, h, args.iters, shift=lam_max, seed=2)
        c = {e: rand_curv(f, theta, e * tn, args.n_rand) for e in args.eps}
        c0 = c[args.eps[0]]
        res[tag] = dict(lam_max=lam_max, lam_min=lam_min, c_rand=c0,
                        stiffness=lam_max / c0 if c0 > 0 else float("nan"),
                        nonconvex=lam_min / lam_max,
                        c_rand_by_eps={str(e): v for e, v in c.items()},
                        kink=c[args.eps[-1]] / c0 if c0 > 0 else float("nan"))
        print(name, tag, json.dumps(res[tag]), flush=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
