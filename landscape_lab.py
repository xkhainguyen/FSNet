"""Interactive loss-landscape helpers for landscape_lab.ipynb.

Pick models, directions, scales, losses and data split; grids are cached, so
re-plotting with a different loss, rho or normalization is instant.

Directions
  ("random", seed)        Gaussian, filter-normalized per output row, biases zeroed
  ("eig", path)           a saved flat vector (e.g. top Hessian eigenvector, hessian_spectrum.py --save_vec)
  ("toward", other)       theta_other - theta (points at another model)
Every direction can be rescaled with `scale` (multiplies its length).
A plane through three models is also available (Lab.plane).

Losses (Lab.loss): "own" (what the model was trained on, rebuilt from its config),
"merit", "merit_fs", or a dict of component weights, e.g. {"huber": 100, "viol_l1": 10}.
Components per grid point: huber, pen2, obj, viol_l1 (raw output) and
huber_fs, obj_fs, viol_l1_fs, dist_fs (after the FS layer; only if fs=True).
"""
import os

# Cap BLAS / OpenMP threads before numpy, scipy or torch start their pools: on a login node
# they default to one thread per node core (128) on a 4-CPU slice and crawl (or hit NPROC).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("LAB_THREADS", "4"))

import hashlib
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from compute_landscape_compare import (load_problem, load_model, random_direction, flat, unflat,
                                       set_weights, eval_components, COMPONENTS)

MERIT_W = 1e5
CACHE_DIR = "figures/landscape/lab_cache"

if not torch.cuda.is_available():
    # login-node slices get 4 CPUs; torch's default (all node cores) oversubscribes and crawls
    torch.set_num_threads(int(os.environ.get("LAB_THREADS", 4)))
FS_KW = dict(val_tol=1e-9, memory=30, max_iter=50, scale=1000)


class Lab:
    def __init__(self, split="test", n_eval=500, batch_size=500, cache_dir=CACHE_DIR):
        self.prob = load_problem()
        ds = self.prob.train_dataset if split == "train" else self.prob.test_dataset
        X, Y = ds.tensors
        self.X, self.Y = X[:n_eval].to(self.prob.device), Y[:n_eval].to(self.prob.device)
        self.split, self.batch_size = split, batch_size
        self.models, self._cache = {}, {}
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        print(f"device: {self.prob.device}, torch threads: {torch.get_num_threads()}")

    # ---------------------------------------------------------------- models
    def add(self, name, ckpt):
        model, cfg = load_model(self.prob, ckpt)
        self.models[name] = dict(model=model, cfg=cfg, ckpt=ckpt, w0=[p.detach().clone() for p in model.parameters()])
        return self

    def add_from_json(self, path="figures/landscape/fig1/ckpts.json"):
        for name, v in json.load(open(path)).items():
            self.add(name, v["ckpt"])
        return self

    def describe(self, name):
        cfg = self.models[name]["cfg"]
        mc = cfg[cfg["method"]]
        return f"{name}: {cfg['method']}, rho={mc.get('eq_pen_weight')}, pen={mc.get('pen_type', 'l2')}, lr={mc.get('lr')}"

    # ----------------------------------------------------------------- cache
    def _disk_key(self, key):
        # checkpoint paths (not model names) identify the weights on disk
        parts = [self.models[k]["ckpt"] if isinstance(k, str) and k in self.models else k for k in key]
        return os.path.join(self.cache_dir, hashlib.md5(repr(parts).encode()).hexdigest() + ".npz")

    def _load(self, key):
        if key in self._cache:
            return self._cache[key]
        if self.cache_dir:
            f = self._disk_key(key)
            if os.path.exists(f):
                d = dict(np.load(f, allow_pickle=True))
                Z = {k: d[k] for k in COMPONENTS}
                out = (d["xs"], d["ys"], Z) + ((d["pts"],) if "pts" in d else ())
                self._cache[key] = out
                return out
        return None

    def _store(self, key, out):
        self._cache[key] = out
        if self.cache_dir:
            extra = {"pts": out[3]} if len(out) > 3 else {}
            np.savez(self._disk_key(key), xs=out[0], ys=out[1], **out[2], **extra)
        return out

    # ------------------------------------------------------------ directions
    def direction(self, name, spec, scale=1.0):
        kind, arg = spec
        params = list(self.models[name]["model"].parameters())
        if kind == "random":
            d = random_direction(params, int(arg))
        elif kind == "eig":
            ref = flat(random_direction(params, 0)).norm()  # same length as a random direction
            v = torch.load(arg).to(params[0].device, params[0].dtype)
            d = unflat(v * ref / v.norm(), params)
        elif kind == "toward":
            d = [a - b for a, b in zip(self.models[arg]["w0"], self.models[name]["w0"])]
        else:
            raise ValueError(kind)
        return [scale * t for t in d]

    # ------------------------------------------------------------------ grid
    def grid(self, name, dx_spec=("random", 5), dy_spec=("random", 6), radius=0.5, n=21,
             dx_scale=1.0, dy_scale=1.0, fs=False, verbose=True):
        """Loss components on theta* + a dx + b dy, a, b in [-radius, radius]."""
        key = (name, dx_spec, dy_spec, float(radius), int(n), float(dx_scale), float(dy_scale), fs, self.split, len(self.X))
        hit = self._load(key)
        if hit is not None:
            return hit
        m = self.models[name]
        dx, dy = self.direction(name, dx_spec, dx_scale), self.direction(name, dy_spec, dy_scale)
        xs = ys = np.linspace(-radius, radius, n)
        Z = {k: np.full((n, n), np.nan) for k in COMPONENTS}
        for j, b in enumerate(ys):
            for i, a in enumerate(xs):
                set_weights(m["model"], m["w0"], dx, dy, float(a), float(b))
                for k, v in eval_components(m["model"], self.prob, self.X, self.Y,
                                            FS_KW if fs else None, self.batch_size).items():
                    Z[k][j, i] = v
            if verbose:
                print(f"\r{name}: row {j + 1}/{n}", end="", flush=True)
        set_weights(m["model"], m["w0"], dx, dy, 0.0, 0.0)
        if verbose:
            print()
        return self._store(key, (xs, ys, Z))

    def plane(self, names, n=31, margin=0.5, fs=False, verbose=True):
        """Grid on the plane through three models (theta_0 at (0,0), theta_1 at (1,0))."""
        key = ("plane",) + tuple(names) + (int(n), float(margin), fs, self.split, len(self.X))
        hit = self._load(key)
        if hit is not None:
            return hit
        ths = [flat(self.models[k]["w0"]) for k in names]
        u = ths[1] - ths[0]
        s = u.norm()
        u = u / s
        v = ths[2] - ths[0]
        v = v - (v @ u) * u
        v = v / v.norm()
        pts = np.array([[((t - ths[0]) @ u / s).item(), ((t - ths[0]) @ v / s).item()] for t in ths])
        lo, hi = pts.min(0) - margin, pts.max(0) + margin
        xs, ys = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
        m = self.models[names[0]]
        like = m["w0"]
        dx, dy = unflat(u * s, like), unflat(v * s, like)
        Z = {k: np.full((n, n), np.nan) for k in COMPONENTS}
        for j, b in enumerate(ys):
            for i, a in enumerate(xs):
                set_weights(m["model"], like, dx, dy, float(a), float(b))
                for k, val in eval_components(m["model"], self.prob, self.X, self.Y,
                                              FS_KW if fs else None, self.batch_size).items():
                    Z[k][j, i] = val
            if verbose:
                print(f"\rplane: row {j + 1}/{n}", end="", flush=True)
        set_weights(m["model"], like, dx, dy, 0.0, 0.0)
        if verbose:
            print()
        return self._store(key, (xs, ys, Z, pts))

    # ------------------------------------------------------------------ loss
    def own_weights(self, name):
        cfg = self.models[name]["cfg"]
        mc, method = cfg[cfg["method"]], cfg["method"]
        rho, pen = mc["eq_pen_weight"], ("viol_l1" if mc.get("pen_type", "l2") == "l1" else "pen2")
        first = {"sup_pen": "huber", "sup_pen_fs": "huber_fs", "penalty": "obj"}[method]
        w = {first: 100.0 if first.startswith("huber") else mc.get("obj_weight", 1.0)}
        w[pen] = w.get(pen, 0.0) + rho
        return w

    def loss(self, Z, spec, name=None):
        if spec == "own":
            spec = self.own_weights(name)
        elif spec == "merit":
            spec = {"obj": 1.0, "viol_l1": MERIT_W}
        elif spec == "merit_fs":
            spec = {"obj_fs": 1.0, "viol_l1_fs": MERIT_W}
        out = sum(w * Z[k] for k, w in spec.items())
        if np.isnan(out).all():
            raise ValueError(f"loss {spec} needs FS components: recompute the grid with fs=True")
        return out


# ------------------------------------------------------------------- plots
def transform(Z, norm="minmax", log=True, floor=1e-3):
    """minmax: (Z - min) / range; rise: (Z - Z_center) / |Z_center|; none: raw."""
    jc, ic = Z.shape[0] // 2, Z.shape[1] // 2
    if norm == "minmax":
        T = (Z - Z.min()) / (Z.max() - Z.min() + 1e-30)
        return np.log10(T + floor) if log else T
    if norm == "rise":
        T = (Z - Z[jc, ic]) / abs(Z[jc, ic])
        return np.sign(T) * np.log10(1 + np.abs(T)) if log else T
    return np.log10(Z - Z.min() + floor * (Z.max() - Z.min())) if log else Z


def show(panels, style=("3d", "contour"), norm="minmax", log=True, shared_z=True, elev=35, azim=190,
         clip_pct=None, points=None, figsize_per=4.2):
    """panels: list of (title, xs, ys, Z). points: optional list of (x, y, label) marked on every panel."""
    T = [transform(np.minimum(Z, np.percentile(Z, clip_pct)) if clip_pct else Z, norm, log) for _, _, _, Z in panels]
    lo, hi = min(t.min() for t in T), max(t.max() for t in T)
    rows = len(style)
    fig = plt.figure(figsize=(figsize_per * len(panels), figsize_per * rows))
    for n, ((title, xs, ys, Z), t) in enumerate(zip(panels, T)):
        X, Y = np.meshgrid(xs, ys)
        zlo, zhi = (lo, hi) if shared_z else (t.min(), t.max())
        j, i = np.unravel_index(np.argmin(Z), Z.shape)
        marks = points or [(0.0, 0.0, "")]
        for r, st in enumerate(style):
            k = r * len(panels) + n + 1
            if st == "3d":
                ax = fig.add_subplot(rows, len(panels), k, projection="3d")
                ax.plot_surface(X, Y, t, cmap=cm.GnBu_r, vmin=zlo, vmax=zhi, shade=False, linewidth=0,
                                antialiased=True, rstride=1, cstride=1)
                for px, py, _ in marks:
                    ii, jj = np.argmin(np.abs(xs - px)), np.argmin(np.abs(ys - py))
                    ax.scatter([xs[ii]], [ys[jj]], [t[jj, ii]], color="r", s=40, depthshade=False)
                ax.set_zlim(zlo, zhi), ax.set_axis_off(), ax.view_init(elev=elev, azim=azim)
                ax.set_box_aspect(None, zoom=1.1)
            elif st == "contour":
                ax = fig.add_subplot(rows, len(panels), k)
                lv = np.linspace(zlo, zhi, 25)
                ax.contourf(X, Y, t, levels=lv, cmap="GnBu_r", extend="both")
                ax.contour(X, Y, t, levels=lv, colors="k", linewidths=0.3, alpha=0.4)
                for px, py, lab in marks:
                    ax.plot(px, py, "o", mfc="r", mec="k", ms=7)
                    if lab:
                        ax.annotate(lab, (px, py), textcoords="offset points", xytext=(4, 4), fontsize=8)
                ax.plot(xs[i], ys[j], "X", color="w", mec="k", ms=10)
                ax.set_aspect("equal")
            elif st == "slice":
                ax = fig.add_subplot(rows, len(panels), k)
                ax.plot(xs, t[len(ys) // 2, :], label="x slice")
                ax.plot(ys, t[:, len(xs) // 2], label="y slice")
                ax.set_ylim(zlo, zhi), ax.legend(fontsize=8)
            if r == 0:
                ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.show()
    return fig


def load_npz(path, prefix, lab=None, loss_spec="merit", name=None):
    """Browse precomputed compute_landscape_compare.py outputs without recomputing."""
    d = dict(np.load(path, allow_pickle=True))
    Z = {k: d[f"{prefix}/{k}"] for k in d["components"]}
    out = (Lab.loss(lab, Z, loss_spec, name) if lab else sum(w * Z[k] for k, w in loss_spec.items()))
    return d["xs"], d["ys"], out, d.get("points")
