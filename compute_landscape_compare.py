"""Loss landscapes of several trained models under several losses.

Every grid point stores per-sample means of the loss components (label
error, squared penalty, objective, L1 violation; raw and after the FSNet
feasibility layer). Any loss of the form a*huber + b*obj + rho*pen2 can then be
rebuilt offline for any rho, so all models can be compared under all losses
on the same grid.

Modes
  random: for each model, a plane through its weights spanned by two
          filter-normalized random directions (Li et al., 2018). The random
          seeds are shared, so every model sees the same raw directions.
  plane:  one plane through three checkpoints (theta_0, theta_1, theta_2),
          Gram-Schmidt basis, coordinates in units of ||theta_1 - theta_0||.
          theta_0 sits at (0, 0) and theta_1 at (1, 0).

Example
  python compute_landscape_compare.py --mode random \
      --ckpt sl_small=path/model.pt --ckpt ssl_small=path/model.pt \
      --out figures/landscape_random.npz
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch

from utils.optimization_utils import nonsmooth_nonconvexSOCPProblem
from utils.lbfgs import nondiff_lbfgs_solve
from utils.trainer import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

COMPONENTS = ["huber", "pen2", "obj", "viol_l1", "huber_fs", "obj_fs", "viol_l1_fs", "dist_fs"]


def huber(x, delta=1e-1):
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * x.pow(2) / delta, ax - 0.5 * delta)


def load_problem(prob_size=(100, 50, 50, 10000), train_size=7000, val_size=1000, test_size=2000):
    path = os.path.join(
        "datasets", "nonsmooth_nonconvex", "socp",
        f"random2025_socp_dataset_var{prob_size[0]}_ineq{prob_size[1]}_eq{prob_size[2]}_ex{prob_size[3]}",
    )
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    prob = nonsmooth_nonconvexSOCPProblem(dataset, train_size, val_size, test_size, 2025, 0)
    prob.device = DEVICE
    for attr in dir(prob):
        var = getattr(prob, attr)
        if torch.is_tensor(var):
            try:
                setattr(prob, attr, var.to(DEVICE))
            except AttributeError:
                pass
    return prob


def load_model(prob, ckpt_path):
    content = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = content["config"]
    model = create_model(prob, cfg["method"], cfg)
    model.load_state_dict(content["model_state_dict"])
    model.eval()
    return model, cfg


# -----------------------------------------------------------------------------
# Directions
# -----------------------------------------------------------------------------
def random_direction(params, seed):
    """Gaussian direction, filter-normalized per output row; biases zeroed."""
    g = torch.Generator(device=params[0].device)
    g.manual_seed(seed)
    direction = []
    for w in params:
        d = torch.randn(w.shape, device=w.device, dtype=w.dtype, generator=g)
        if d.dim() <= 1:
            d.zero_()
        else:
            d.mul_(w.norm(dim=1, keepdim=True) / (d.norm(dim=1, keepdim=True) + 1e-10))
        direction.append(d)
    return direction


def flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def unflat(v, like):
    out, i = [], 0
    for t in like:
        out.append(v[i:i + t.numel()].view_as(t))
        i += t.numel()
    return out


@torch.no_grad()
def set_weights(model, w0, dx, dy, a, b):
    for p, w, u, v in zip(model.parameters(), w0, dx, dy):
        p.copy_(w + a * u + b * v)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def eval_components(model, prob, X, Y_label, fs_kwargs, batch_size):
    """Per-sample means of every loss component over (X, Y_label)."""
    sums = {k: 0.0 for k in COMPONENTS}
    n = X.shape[0]
    for s in range(0, n, batch_size):
        x, yl = X[s:s + batch_size], Y_label[s:s + batch_size]
        with torch.no_grad():
            y = prob.scale(model(x))
            eq, ineq = prob.eq_resid(x, y), prob.ineq_resid(x, y)
            sums["huber"] += huber(y - yl).mean(dim=1).sum().item()
            sums["pen2"] += (eq.square().sum(1) + ineq.square().sum(1)).sum().item()
            sums["obj"] += prob.obj_fn(y).sum().item()
            sums["viol_l1"] += (eq.abs().sum(1) + ineq.abs().sum(1)).sum().item()
        if fs_kwargs is not None:
            y_fs = nondiff_lbfgs_solve(x, y.detach(), prob, **fs_kwargs).detach()
            with torch.no_grad():
                eq, ineq = prob.eq_resid(x, y_fs), prob.ineq_resid(x, y_fs)
                sums["huber_fs"] += huber(y_fs - yl).mean(dim=1).sum().item()
                sums["obj_fs"] += prob.obj_fn(y_fs).sum().item()
                sums["viol_l1_fs"] += (eq.abs().sum(1) + ineq.abs().sum(1)).sum().item()
                sums["dist_fs"] += (y_fs - y).square().sum(1).sum().item()
    return {k: v / n for k, v in sums.items()}


def eval_grid(model, prob, X, Y_label, w0, dx, dy, xs, ys, fs_kwargs, batch_size, tag):
    Z = {k: np.full((len(ys), len(xs)), np.nan) for k in COMPONENTS}
    total, k, t0 = len(xs) * len(ys), 0, time.time()
    for j, b in enumerate(ys):
        for i, a in enumerate(xs):
            set_weights(model, w0, dx, dy, float(a), float(b))
            comp = eval_components(model, prob, X, Y_label, fs_kwargs, batch_size)
            for key, v in comp.items():
                Z[key][j, i] = v
            k += 1
            if k % max(1, total // 20) == 0:
                print(f"[{tag}] {k}/{total}  {time.time() - t0:.0f}s", flush=True)
    set_weights(model, w0, dx, dy, 0.0, 0.0)
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "plane"], default="random")
    ap.add_argument("--ckpt", action="append", required=True, help="name=path/to/model.pt (repeatable)")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--xnum", type=int, default=41)
    ap.add_argument("--range", type=float, nargs=2, default=[-1.0, 1.0],
                    help="random mode: alpha/beta range; plane mode: margin is added around the 3 points")
    ap.add_argument("--plane_margin", type=float, default=0.5)
    ap.add_argument("--dir_seeds", type=int, nargs=2, default=[5, 6])
    ap.add_argument("--no_fs", action="store_true", help="skip the feasibility layer (much cheaper)")
    ap.add_argument("--dir_vec", default=None, help="random mode: .pt flat vector used as the x direction")
    ap.add_argument("--fs_max_iter", type=int, default=50)
    ap.add_argument("--fs_tol", type=float, default=1e-9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names, paths = zip(*[c.split("=", 1) for c in args.ckpt])
    prob = load_problem()
    dataset = prob.train_dataset if args.split == "train" else prob.test_dataset
    X, Y_label = dataset.tensors
    X, Y_label = X[:args.n_eval].to(DEVICE), Y_label[:args.n_eval].to(DEVICE)

    fs_kwargs = None if args.no_fs else dict(
        val_tol=args.fs_tol, memory=30, max_iter=args.fs_max_iter, scale=1000
    )

    out = {"mode": args.mode, "names": np.array(names), "paths": np.array(paths),
           "components": np.array(COMPONENTS), "split": args.split, "n_eval": args.n_eval}

    if args.mode == "random":
        xs = np.linspace(args.range[0], args.range[1], args.xnum)
        ys = xs.copy()
        out["xs"], out["ys"] = xs, ys
        for name, path in zip(names, paths):
            model, _ = load_model(prob, path)
            params = list(model.parameters())
            w0 = [p.detach().clone() for p in params]
            dx = random_direction(params, args.dir_seeds[0])
            dy = random_direction(params, args.dir_seeds[1])
            if args.dir_vec:
                # x axis = a given direction (e.g. top Hessian eigenvector), rescaled to the
                # norm of the random y direction so both axes span the same weight distance
                v = torch.load(args.dir_vec).to(params[0].device, params[0].dtype)
                v = v * (flat(dy).norm() / v.norm())
                dx = unflat(v, params)
            Z = eval_grid(model, prob, X, Y_label, w0, dx, dy, xs, ys, fs_kwargs, args.batch_size, name)
            for key, v in Z.items():
                out[f"{name}/{key}"] = v
    else:
        assert len(names) == 3, "plane mode needs exactly 3 checkpoints"
        models = [load_model(prob, p)[0] for p in paths]
        like = [p.detach().clone() for p in models[0].parameters()]
        th = [flat([p.detach() for p in m.parameters()]) for m in models]
        u = th[1] - th[0]
        scale = u.norm()
        u = u / scale
        v = th[2] - th[0]
        v = v - (v @ u) * u
        v = v / v.norm()
        # Coordinates of the three points, in units of ||theta_1 - theta_0||.
        pts = np.array([[((t - th[0]) @ u / scale).item(), ((t - th[0]) @ v / scale).item()] for t in th])
        lo, hi = pts.min(0) - args.plane_margin, pts.max(0) + args.plane_margin
        xs = np.linspace(lo[0], hi[0], args.xnum)
        ys = np.linspace(lo[1], hi[1], args.xnum)
        out["xs"], out["ys"], out["points"] = xs, ys, pts
        out["plane_scale"] = scale.item()
        dx, dy = unflat(u * scale, like), unflat(v * scale, like)
        Z = eval_grid(models[0], prob, X, Y_label, like, dx, dy, xs, ys, fs_kwargs, args.batch_size, "plane")
        for key, val in Z.items():
            out[f"plane/{key}"] = val

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
