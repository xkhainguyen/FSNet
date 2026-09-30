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
from utils.lbfgs import nondiff_lbfgs_solve, LBFGSConfig, compute_gamma, _search_direction
from utils.trainer import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

COMPONENTS = ["huber", "pen2", "obj", "viol_l1", "huber_fs", "obj_fs", "viol_l1_fs", "dist_fs", "pen2_gated"]
# pen2_gated: the FSNet-style gated penalty exactly as in Trainer._sup_pen_fs_loss (pen_gate 1e3):
# per evaluation batch, squared raw eq + ineq violation, counted only if the batch-mean squared eq
# or ineq violation is >= GATE; averaged over samples like the other components.
GATE = 1e3


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
            eq2, in2 = eq.square().sum(1), ineq.square().sum(1)
            on = bool(eq2.mean() >= GATE or in2.mean() >= GATE)
            sums["pen2_gated"] += float(on) * (eq2 + in2).sum().item()
        if fs_kwargs is not None:
            y_fs = nondiff_lbfgs_solve(x, y.detach(), prob, **fs_kwargs).detach()
            with torch.no_grad():
                eq, ineq = prob.eq_resid(x, y_fs), prob.ineq_resid(x, y_fs)
                sums["huber_fs"] += huber(y_fs - yl).mean(dim=1).sum().item()
                sums["obj_fs"] += prob.obj_fn(y_fs).sum().item()
                sums["viol_l1_fs"] += (eq.abs().sum(1) + ineq.abs().sum(1)).sum().item()
                sums["dist_fs"] += (y_fs - y).square().sum(1).sum().item()
    return {k: v / n for k, v in sums.items()}


def grouped_nondiff_lbfgs(x, y_init, prob, **fs_kwargs):
    """nondiff_lbfgs_solve run on G independent batches at once; x (G, B, dx), y_init (G, B, n).

    Identical to calling nondiff_lbfgs_solve on each group separately: every group has its own
    batch-mean objective, its own scalar backtracking step and its own stopping test, and a
    converged group is frozen (the original breaks out of its loop). The L-BFGS direction and
    gamma are per row in the original, so they are computed on the flattened (G * B, n) rows.
    """
    cfg = LBFGSConfig(**fs_kwargs)
    G, B, n = y_init.shape
    xf = x.reshape(G * B, -1)

    def obj(y, gidx=None):  # (g, B, n) -> (g,); gidx: which groups y holds (default: all)
        g_ = y.shape[0]
        xs_ = xf if gidx is None else x[gidx].reshape(g_ * B, -1)
        yf = y.reshape(g_ * B, n)
        eq = (prob.eq_resid(xs_, yf) ** 2).sum(1).view(g_, B).mean(1)
        ineq = (prob.ineq_resid(xs_, yf) ** 2).sum(1).view(g_, B).mean(1)
        return cfg.scale * (eq + ineq)

    def f_and_g(y):
        y = y.detach().requires_grad_(True)
        f = obj(y)
        g, = torch.autograd.grad(f.sum(), y)
        return f.detach(), g.detach()

    y = y_init.detach().clone()
    f_val, g = f_and_g(y)
    S_hist = torch.zeros(cfg.memory, G * B, n, device=y.device, dtype=y.dtype)
    Y_hist = torch.zeros_like(S_hist)
    hist_len = hist_ptr = 0
    active = torch.ones(G, dtype=torch.bool, device=y.device)
    for _ in range(cfg.max_iter):
        conv = (f_val / cfg.scale < cfg.val_tol) | (g.norm(dim=2) < cfg.grad_tol).all(1)
        active = active & ~conv
        if not active.any():
            break
        gf = g.reshape(G * B, n)
        if hist_len > 0:
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=y.device)) % cfg.memory
            S, Yh = S_hist[idx], Y_hist[idx]
            d = _search_direction(gf, S, Yh, compute_gamma(S, Yh)).view(G, B, n)
        else:
            d = -0.1 * g
        d = torch.where(active[:, None, None], d, torch.zeros_like(d))  # frozen groups do not move
        # per-group backtracking line search (same rule as utils.lbfgs._backtracking_line_search)
        dir_deriv = (g * d).sum((1, 2))
        # Each trial is evaluated only on the groups still backtracking; a group's step is halved
        # after each failed trial, max_ls_iter trials at most, exactly as in the original.
        step = torch.ones(G, device=y.device, dtype=y.dtype)
        todo = torch.nonzero(active).squeeze(1)
        with torch.no_grad():
            for _ in range(cfg.max_ls_iter):
                if todo.numel() == 0:
                    break
                st = step[todo]
                ok = obj(y[todo] + st[:, None, None] * d[todo], todo) <= f_val[todo] + cfg.c * st * dir_deriv[todo]
                bad = todo[~ok]
                step[bad] = step[bad] * cfg.rho_ls
                todo = bad
        y_next = torch.where(active[:, None, None], y + step[:, None, None] * d, y)
        f_next, g_next = f_and_g(y_next)
        S_hist[hist_ptr] = (y_next - y).reshape(G * B, n)
        Y_hist[hist_ptr] = (g_next - g).reshape(G * B, n)
        hist_ptr = (hist_ptr + 1) % cfg.memory
        hist_len = min(hist_len + 1, cfg.memory)
        keep = active[:, None, None]
        y = y_next
        f_val = torch.where(active, f_next, f_val)
        g = torch.where(keep, g_next, g)
    return y


def eval_grid_grouped(model, prob, X, Y_label, w0, dx, dy, pts_ab, fs_kwargs, batch_size):
    """Components at several grid points at once (one grouped FS solve). pts_ab: [(a, b), ...]."""
    setters = [lambda a=a, b=b: set_weights(model, w0, dx, dy, float(a), float(b)) for a, b in pts_ab]
    return eval_points_grouped(model, prob, X, Y_label, setters, fs_kwargs, batch_size)


def eval_points_grouped(model, prob, X, Y_label, setters, fs_kwargs, batch_size):
    """Components at several weight settings at once; setters[p]() loads the p-th weights into model."""
    xb = [X[s:s + batch_size] for s in range(0, X.shape[0], batch_size)]
    lb = [Y_label[s:s + batch_size] for s in range(0, X.shape[0], batch_size)]
    assert all(len(v) == len(xb[0]) for v in xb), "n_eval must be a multiple of batch_size"
    ys = []
    with torch.no_grad():
        for setter in setters:
            setter()
            ys.append(torch.stack([prob.scale(model(x)) for x in xb]))  # (nb, B, n)
    Yp = torch.stack(ys)  # (P, nb, B, n)
    P, nb, B, n = Yp.shape
    Xg = torch.stack(xb)[None].expand(P, -1, -1, -1).reshape(P * nb, B, -1)
    Lg = torch.stack(lb)[None].expand(P, -1, -1, -1).reshape(P * nb, B, -1)
    Yg = Yp.reshape(P * nb, B, n)
    Yfs = grouped_nondiff_lbfgs(Xg, Yg, prob, **fs_kwargs) if fs_kwargs is not None else None
    out = []
    with torch.no_grad():
        xf, yf, lf = Xg.reshape(P * nb * B, -1), Yg.reshape(-1, n), Lg.reshape(P * nb * B, -1)
        per = lambda v: v.view(P, nb * B).mean(1)  # per-sample mean for each grid point
        eq, ineq = prob.eq_resid(xf, yf), prob.ineq_resid(xf, yf)
        eq2, in2 = eq.square().sum(1).view(P * nb, B), ineq.square().sum(1).view(P * nb, B)
        on = ((eq2.mean(1) >= GATE) | (in2.mean(1) >= GATE)).to(eq2.dtype)[:, None]  # per eval batch
        comp = {"huber": per(huber(yf - lf).mean(1)), "pen2": per(eq.square().sum(1) + ineq.square().sum(1)),
                "pen2_gated": per((on * (eq2 + in2)).reshape(-1)),
                "obj": per(prob.obj_fn(yf)), "viol_l1": per(eq.abs().sum(1) + ineq.abs().sum(1))}
        if Yfs is not None:
            yfs = Yfs.reshape(-1, n)
            eq, ineq = prob.eq_resid(xf, yfs), prob.ineq_resid(xf, yfs)
            comp.update({"huber_fs": per(huber(yfs - lf).mean(1)), "obj_fs": per(prob.obj_fn(yfs)),
                         "viol_l1_fs": per(eq.abs().sum(1) + ineq.abs().sum(1)),
                         "dist_fs": per((yfs - yf).square().sum(1))})
        for p in range(P):
            out.append({k: v[p].item() for k, v in comp.items()})
    return out


def eval_grid(model, prob, X, Y_label, w0, dx, dy, xs, ys, fs_kwargs, batch_size, tag, shard=(0, 1), group=1):
    """shard (k, n): evaluate only rows j with j % n == k (others stay NaN; merge shards afterwards)."""
    Z = {k: np.full((len(ys), len(xs)), np.nan) for k in COMPONENTS}
    rows = [j for j in range(len(ys)) if j % shard[1] == shard[0]]
    total, k, t0 = len(xs) * len(rows), 0, time.time()
    if group > 1:  # several grid points per grouped FS solve (same numbers, much faster)
        cells = [(j, i) for j in rows for i in range(len(xs))]
        for c0 in range(0, len(cells), group):
            chunk = cells[c0:c0 + group]
            res = eval_grid_grouped(model, prob, X, Y_label, w0, dx, dy, [(xs[i], ys[j]) for j, i in chunk],
                                    fs_kwargs, batch_size)
            for (j, i), comp in zip(chunk, res):
                for key, v in comp.items():
                    Z[key][j, i] = v
            k += len(chunk)
            if (c0 // group) % max(1, total // group // 20) == 0:
                print(f"[{tag}] {k}/{total}  {time.time() - t0:.0f}s", flush=True)
        set_weights(model, w0, dx, dy, 0.0, 0.0)
        return Z
    for j in rows:
        b = ys[j]
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
    ap.add_argument("--shard", default="0/1", help="k/n: evaluate rows j %% n == k only (plane mode)")
    ap.add_argument("--group", type=int, default=1, help="grid points per grouped FS solve (1 = original loop)")
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
        Z = eval_grid(models[0], prob, X, Y_label, like, dx, dy, xs, ys, fs_kwargs, args.batch_size, "plane",
                       tuple(int(v) for v in args.shard.split("/")), args.group)
        for key, val in Z.items():
            out[f"plane/{key}"] = val

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
