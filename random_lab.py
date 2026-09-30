"""Random-direction landscapes around one trained network per method (Li et al. 2018), with a disk cache.

A plane is fixed by (method, seed, dir seeds (a, b), radius, scale (sa, sb), grid n, n_eval):
theta + x * sa * d_a + y * sb * d_b for x, y in [-radius, radius] (axes are shown in units of d_a and
d_b, i.e. x * sa and y * sb), with d_a, d_b filter-normalized Gaussian directions (per output row,
biases zeroed) drawn on the CPU from torch seeds a and b, so every method and every device uses the
same draw. The loss is each method's own training loss on held-out instances (compute_landscape_compare
components, fig1_common.own_loss); FS methods go through the FS layer (grouped solver on GPU).

In a notebook: see random_lab.ipynb. From the shell (what the submitted jobs run):
  python random_lab.py --spec M3f:3:0,1:1.0 --spec M3r1:1:0,1:1.0:2,0.5 --n 61   (last field: scale, optional)
"""
import argparse
import os
import subprocess

import numpy as np
import torch

CACHE = "figures/landscape/fig1/random_cache"
ALIGNED = "figures/landscape/fig1/aligned"
FS_KW = dict(val_tol=1e-9, memory=30, max_iter=50, scale=1000)
COLLAPSED = {"M3r1": {1, 3, 8}, "M3r08": {2, 5, 6, 7, 8}}  # merit after FS > 10 (m3_candidates.py)


def needs_fs(m):
    return m.startswith("M3")


def cache_path(m, seed, dirs, radius, n, n_eval, scale=(1.0, 1.0)):
    sc = "" if tuple(map(float, scale)) == (1.0, 1.0) else f"_sc{scale[0]:g}x{scale[1]:g}"
    return f"{CACHE}/{m}_s{seed}_d{dirs[0]}-{dirs[1]}_r{radius:g}{sc}_n{n}_e{n_eval}.npz"


def cpu_direction(params, seed):
    """Same as compute_landscape_compare.random_direction, but always drawn on the CPU."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for w in params:
        d = torch.randn(w.shape, dtype=w.dtype, generator=g).to(w.device)
        if d.dim() <= 1:
            d.zero_()
        else:
            d.mul_(w.norm(dim=1, keepdim=True) / (d.norm(dim=1, keepdim=True) + 1e-10))
        out.append(d)
    return out


def compute(m, seed, dirs, radius, n=61, n_eval=1000, group=16, scale=(1.0, 1.0)):
    """Evaluate one plane and write it to the cache (returns the path)."""
    import compute_landscape_compare as C
    path = cache_path(m, seed, dirs, radius, n, n_eval, scale)
    if os.path.exists(path):
        return path
    prob = C.load_problem()
    X, Y = [t[:n_eval].to(C.DEVICE) for t in prob.test_dataset.tensors]
    net, _ = C.load_model(prob, f"{ALIGNED}/{m}_s{seed}.pt")
    params = list(net.parameters())
    w0 = [p.detach().clone() for p in params]
    dx, dy = cpu_direction(params, dirs[0]), cpu_direction(params, dirs[1])
    xs = np.linspace(-radius * scale[0], radius * scale[0], n)  # coordinates in units of d_a, d_b
    ys = np.linspace(-radius * scale[1], radius * scale[1], n)
    fs = FS_KW if needs_fs(m) else None
    g = group if (fs is not None and C.DEVICE.type == "cuda") else 1
    Z = C.eval_grid(net, prob, X, Y, w0, dx, dy, xs, ys, fs, 500, f"{m}_s{seed}", (0, 1), g)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, xs=xs, ys=ys, points=np.zeros((1, 2)), components=np.array(list(Z)),
             method=m, seed=seed, dirs=np.array(dirs), radius=radius, scale=np.array(scale), n_eval=n_eval,
             **{f"plane/{k}": v for k, v in Z.items()})
    return path


def submit(specs, n=61, n_eval=1000, gpu="h200:1"):
    """One SLURM job computing every missing spec (m, seed, (a, b), radius, (sa, sb)); returns the job id
    or None if everything is cached."""
    todo = [s for s in specs if not os.path.exists(cache_path(*s[:4], n, n_eval, s[4]))]
    if not todo:
        return None
    args = " ".join(f"--spec {m}:{sd}:{a},{b}:{r:g}:{sa:g},{sb:g}" for m, sd, (a, b), r, (sa, sb) in todo)
    pre = ("source ~/.bashrc; conda activate ml4opt; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; "
           f"cd {os.getcwd()}")
    cmd = ["sbatch", "--parsable", "-J", "randlab", "-p", "pi_donti_gpu,mit_normal_gpu,mit_preemptable", "-G", gpu,
           "--cpus-per-task=4", "--mem=32G", "-t", "02:00:00", "-o", "logs/%x-%j.out", "-e", "logs/%x-%j.err",
           "--wrap", f"{pre}; python random_lab.py {args} --n {n} --n_eval {n_eval}"]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def load(m, seed, dirs, radius, n=61, n_eval=1000, scale=(1.0, 1.0)):
    path = cache_path(m, seed, dirs, radius, n, n_eval, scale)
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None


def surface(d, m, cap=4.0, norm="method"):
    """Panel dict for fig1_plot's draw functions. norm: "method" = relative to the method's best of its
    10 trained networks (seed_losses.py); "center" = relative to the network at the centre."""
    from fig1_common import own_loss
    from fig1_plot import method_ref
    c = {k: d[f"plane/{k}"].astype(float) for k in d["components"]}
    Z = own_loss(m, c)
    xs, ys = d["xs"], d["ys"]
    j0, i0 = int(np.argmin(np.abs(ys))), int(np.argmin(np.abs(xs)))
    ref = method_ref(m, "own") if norm == "method" else Z[j0, i0]
    T = np.log10(Z / ref)
    return dict(method=m, xs=xs, ys=ys, pts=np.zeros((1, 2)), Z=Z, T=T, Tc=np.clip(T, 0, cap),
                idx=[(j0, i0)], minima=(np.array([], int), np.array([], int)))


def as_list(choices):
    """choices as a list of (method, dict(seed=, dirs=(a, b), radius=)); a dict {method: ...} also works."""
    return list(choices.items()) if isinstance(choices, dict) else list(choices)


def specs(choices):
    return [(m, ch["seed"], tuple(ch["dirs"]), float(ch["radius"]), tuple(map(float, ch.get("scale", (1, 1)))))
            for m, ch in as_list(choices)]


def status(job):
    """'pending' / 'running' / 'done' for a job id from submit()."""
    if job is None:
        return "done"
    out = subprocess.run(["squeue", "-j", str(job), "-h", "-o", "%T"], capture_output=True, text=True).stdout.strip()
    return {"PENDING": "pending", "RUNNING": "running", "": "done"}.get(out.split("\n")[0], out.lower())


def figure(choices, n=61, n_eval=1000, cap=4.0, norm="method", elev=42, azim=-65, slice_ymax=6.0, path=None):
    """choices: list of (method, dict(seed=, dirs=(a, b), radius=, scale=(sa, sb) optional)) or a dict.
    Columns = choices; rows =
    3D surface, top view, loss along the first direction through the network (uncapped).
    Planes not cached yet are skipped."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from fig1_plot import draw3d, draw2d, LABEL
    sc = lambda ch: tuple(map(float, ch.get("scale", (1, 1))))
    cols = [(m, ch) for m, ch in as_list(choices)
            if load(m, ch["seed"], tuple(ch["dirs"]), ch["radius"], n, n_eval, sc(ch)) is not None]
    if not cols:
        print("nothing cached yet")
        return None
    fig = plt.figure(figsize=(5.2 * len(cols), 14))
    gs = fig.add_gridspec(3, len(cols), hspace=0.12, wspace=0.08, height_ratios=[1.1, 1, 0.8])
    cs, axes2d = None, []
    for c, (m, ch) in enumerate(cols):
        s = surface(load(m, ch["seed"], tuple(ch["dirs"]), ch["radius"], n, n_eval, sc(ch)), m, cap, norm)
        st = dict(cap=cap, elev=elev, azim=azim, stride=1, mesh=False, minima=False, seed_labels=False,
                  cmap="viridis", xyticks=True, label_size=14,
                  collapsed={m: {0}} if ch["seed"] in COLLAPSED.get(m, ()) else {})
        ax = fig.add_subplot(gs[0, c], projection="3d")
        draw3d(ax, s, st)
        tag = " (collapsed)" if ch["seed"] in COLLAPSED.get(m, ()) else ""
        scl = "" if sc(ch) == (1.0, 1.0) else f", scale {sc(ch)[0]:g} x {sc(ch)[1]:g}"
        ax.set_title(f"{LABEL[m]}\nseed {ch['seed']}{tag}, dirs {tuple(ch['dirs'])}, radius {ch['radius']:g}{scl}",
                     fontsize=11)
        ax2 = fig.add_subplot(gs[1, c])
        cs = draw2d(ax2, s, st)
        ax2.axhline(0, color="w", lw=1.2, ls="--")
        axes2d.append(ax2)
        ax3 = fig.add_subplot(gs[2, c])
        t = s["T"][s["idx"][0][0]]
        ax3.plot(s["xs"], np.minimum(t, slice_ymax), color="k", lw=1.3)
        over = t > slice_ymax
        if over.any():
            ax3.plot(s["xs"][over], np.full(over.sum(), slice_ymax), "^", color="k", ms=4)
        k = "X" if tag else "o"
        ax3.plot(0, t[s["idx"][0][1]], k, mfc="k" if tag else "r", mec="w" if tag else "k", ms=10)
        ax3.set_ylim(-0.1, slice_ymax + 0.2), ax3.set_xlabel(f"direction {ch['dirs'][0]} (filter-normalized)")
        ax3.spines[["top", "right"]].set_visible(False)
        if c == 0:
            ax3.set_ylabel(r"$\log_{10}(L / L_{\mathrm{best}})$ along the dashed line")
    ref = "the method's best trained network" if norm == "method" else "the network at the centre"
    fig.colorbar(cs, ax=axes2d, shrink=0.8, pad=0.01, ticks=range(int(cap) + 1),
                 label=f"log10(L / L at {ref}), held-out")
    fig.legend(handles=[Line2D([], [], ls="", marker="o", mfc="r", mec="k", ms=9, label="trained network"),
                        Line2D([], [], ls="", marker="X", mfc="k", mec="w", ms=11, label="trained, collapsed")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.45, 0.0), frameon=False)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(path)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True, help="method:seed:a,b:radius")
    ap.add_argument("--n", type=int, default=61)
    ap.add_argument("--n_eval", type=int, default=1000)
    args = ap.parse_args()
    for spec in args.spec:
        parts = spec.split(":")
        m, sd, ab, r = parts[:4]
        a, b = (int(v) for v in ab.split(","))
        scale = tuple(float(v) for v in parts[4].split(",")) if len(parts) > 4 else (1.0, 1.0)
        print(compute(m, int(sd), (a, b), float(r), args.n, args.n_eval, scale=scale), flush=True)


if __name__ == "__main__":
    main()
