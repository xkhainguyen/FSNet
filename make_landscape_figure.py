"""Paper figure: training-loss vs merit landscapes around each trained model.

Row 1: each model's own training loss on a random plane around its weights.
Row 2: the task merit M on the same plane (raw merit for soft-constraint
       models, merit after the FS layer for FS models, i.e. what each model
       is judged by at inference).
Both rows share one absolute z scale per row: log10(1 + (L - L0)/|L0|),
the relative rise from the trained weights, so sharper surfaces look steeper.
The trained weights (red) and the grid minimum of M (white x) are marked.

Also writes training curves (own loss and merit per epoch, from results.pkl)
and prints the alignment table: M at the trained weights vs M at its grid
minimum, and where that minimum is.

  python make_landscape_figure.py --tag v2_r0.1_n41 --models sl_small ssl_small sl_high ssl_l1_high sl_fs_small
"""
import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from plot_landscape_compare import build_losses

D = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
PRE, POST = "seed0_nepochs3000_lr0.0003_trainsize7000", "dropout0.0_lrschedcosine_etamin1e-06"

# name: (training-loss key, merit key, run-dir pattern, label, penalty weight, pen type)
MODELS = {
    "sl_small": ("L_sl", "merit", f"sup_pen_{PRE}_obj0.1_eq10.0_ineq10.0_{POST}", r"SL, small $\rho$", 10, "l2"),
    "sl_high": ("L_sl_high", "merit", f"sup_pen_{PRE}_obj0.1_eq100000.0_ineq100000.0_{POST}", r"SL, large $\rho$", 1e5, "l2"),
    "sl_fs_small": ("L_sl_fs", "merit_fs", f"sup_pen_fs_{PRE}_obj0.1_eq10.0_ineq10.0_{POST}", r"SL + FS, small $\rho$", 10, "l2"),
    "ssl_small": ("L_ssl", "merit", f"penalty_{PRE}_obj1.0_eq10.0_ineq10.0_{POST}", r"SSL, small $\rho$", 10, "l2"),
    "sl_l1_small": ("L_sl_l1_small", "merit", f"sup_pen_{PRE}_obj0.1_eq10.0_ineq10.0_penl1_{POST}", r"SL (L1), small $\rho$", 10, "l1"),
    "sl_l1_high": ("L_sl_l1_high", "merit", f"sup_pen_{PRE}_obj0.1_eq100000.0_ineq100000.0_penl1_{POST}", r"SL (L1), large $\rho$", 1e5, "l1"),
    "ssl_l1_small": ("L_ssl_l1_small", "merit", f"penalty_{PRE}_obj1.0_eq10.0_ineq10.0_penl1_{POST}", r"SSL (L1), small $\rho$", 10, "l1"),
    "ssl_l1_high": ("L_ssl_l1_high", "merit", f"penalty_{PRE}_obj1.0_eq100000.0_ineq100000.0_penl1_{POST}", r"SSL (L1), large $\rho$ (= $\mathcal{M}$)", 1e5, "l1"),
    "sl_fs_l1_small": ("L_sl_fs_l1", "merit_fs", f"sup_pen_fs_{PRE}_obj0.1_eq10.0_ineq10.0_penl1_{POST}", r"SL + FS (L1), small $\rho$", 10, "l1"),
}


def load_losses(tag, name):
    d = dict(np.load(f"figures/landscape/random_{tag}_{name}.npz", allow_pickle=True))
    L = build_losses(d, name, 10.0, 1e3, 1e5)
    c = {k: d[f"{name}/{k}"] for k in d["components"]}
    L["L_sl_fs_l1"] = 100 * c["huber_fs"] + 10 * c["viol_l1"]
    return d["xs"], d["ys"], L


def rise(Z):
    jc, ic = Z.shape[0] // 2, Z.shape[1] // 2
    return (Z - Z[jc, ic]) / abs(Z[jc, ic])


def surface(ax, X, Y, Zp, zmin, zmax, marks):
    ax.plot_surface(X, Y, Zp, cmap=cm.GnBu_r, vmin=zmin, vmax=zmax, shade=False,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    for (px, py), style in marks:
        i, j = np.argmin(np.abs(X[0] - px)), np.argmin(np.abs(Y[:, 0] - py))
        ax.scatter([X[0, i]], [Y[j, 0]], [Zp[j, i]], depthshade=False, **style)
    ax.set_zlim(zmin, zmax)
    ax.set_axis_off()
    ax.set_box_aspect(None, zoom=1.15)
    ax.view_init(elev=35, azim=190)


def landscape_figure(tag, models, out):
    xs, ys, rows = None, None, []
    for m in models:
        xs, ys, L = load_losses(tag, m)
        own_key, merit_key = MODELS[m][0], MODELS[m][1]
        rows.append((m, L[own_key], L[merit_key]))
    X, Y = np.meshgrid(xs, ys)
    tr = lambda Z: np.sign(rise(Z)) * np.log10(1 + np.abs(rise(Z)))
    own_T = [tr(o) for _, o, _ in rows]
    mer_T = [tr(mz) for _, _, mz in rows]
    lims = [(min(t.min() for t in T), max(t.max() for t in T)) for T in (own_T, mer_T)]

    fig = plt.figure(figsize=(3.6 * len(models), 7.2))
    for n, (m, own, mer) in enumerate(rows):
        j, i = np.unravel_index(np.argmin(mer), mer.shape)
        marks = [((0, 0), dict(color="r", s=45)), ((xs[i], ys[j]), dict(color="w", marker="X", s=70, edgecolors="k"))]
        ax = fig.add_subplot(2, len(models), n + 1, projection="3d")
        surface(ax, X, Y, own_T[n], *lims[0], marks[:1])
        ax.set_title(MODELS[m][3], fontsize=12, pad=-2)
        ax = fig.add_subplot(2, len(models), len(models) + n + 1, projection="3d")
        surface(ax, X, Y, mer_T[n], *lims[1], marks)
    fig.text(0.005, 0.73, "training loss", rotation=90, fontsize=13, va="center")
    fig.text(0.005, 0.27, r"merit $\mathcal{M}$", rotation=90, fontsize=13, va="center")
    fig.subplots_adjust(wspace=0, hspace=0.02, left=0.02, right=1, top=0.94, bottom=0)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n## {tag}: training loss vs merit on each model's random plane\n")
    print("| model | own-loss rise at edge | merit rise at edge | merit at weights | merit grid min | min at (a, b) | merit gain possible |")
    print("|---|---:|---:|---:|---:|---|---:|")
    for m, own, mer in rows:
        ring = lambda Z: np.concatenate([Z[0], Z[-1], Z[1:-1, 0], Z[1:-1, -1]]).mean()
        c = mer[mer.shape[0] // 2, mer.shape[1] // 2]
        j, i = np.unravel_index(np.argmin(mer), mer.shape)
        print(f"| {m} | {ring(rise(own)):.3g} | {ring(rise(mer)):.3g} | {c:.4g} | {mer.min():.4g} | "
              f"({xs[i]:+.3f}, {ys[j]:+.3f}) | {(c - mer.min()) / abs(c):.1%} |")


def training_curves(models, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for m in models:
        runs = [r for r in sorted(glob.glob(f"{D}/*_MLP_{MODELS[m][2]}")) if os.path.exists(r + "/results.pkl")]
        if not runs:
            continue
        h = pickle.load(open(runs[-1] + "/results.pkl", "rb"))["train_history"]
        viol = np.array([e["eq_violation_l1"] + e["ineq_violation_l1"] for e in h])
        merit = np.array([e["obj"] for e in h]) + 1e5 * viol
        axes[0].plot(viol, label=MODELS[m][3])
        axes[1].plot(merit, label=MODELS[m][3])
    axes[0].set_yscale("log"), axes[0].set_ylabel("train constraint violation (L1, per epoch)")
    axes[1].set_yscale("log"), axes[1].set_ylabel(r"train merit $\mathcal{M}$ (per epoch)")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    fig.savefig(out, dpi=200)
    plt.close(fig)


PLANE_KEYS = [("L_sl", r"SL loss, small $\rho$"), ("L_sl_fs", r"SL + FS loss, small $\rho$"),
              ("L_ssl", r"SSL loss, small $\rho$"),
              ("L_sl_high", r"SL loss, large $\rho$"), ("merit", r"merit $\mathcal{M}$ (large $\rho$)"),
              ("merit_fs", r"merit $\mathcal{M}$ after FS")]


def plane_figure(npz, out, keys=PLANE_KEYS):
    """Every loss on ONE shared plane through three trained models (same xy)."""
    from matplotlib import colors
    d = dict(np.load(npz, allow_pickle=True))
    L = build_losses(d, "plane", 10.0, 1e3, 1e5)
    xs, ys, pts, names = d["xs"], d["ys"], d["points"], list(d["names"])
    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(4.2 * len(keys), 7.6))
    for n, (k, title) in enumerate(keys):
        Z = L[k]
        Zp = Z - Z.min() + 1e-3 * (Z.max() - Z.min())  # floor avoids log needles at the grid minimum
        j, i = np.unravel_index(np.argmin(Z), Z.shape)
        ax = fig.add_subplot(2, len(keys), n + 1)
        top = np.percentile(Zp, 97)  # clip rare FS-failure spikes so the rest of the surface is readable
        Zp = np.minimum(Zp, top)
        lv = np.geomspace(Zp.min(), top, 25)
        ax.contourf(X, Y, Zp, levels=lv, norm=colors.LogNorm(), cmap="GnBu_r")
        ax.contour(X, Y, Zp, levels=lv, colors="k", linewidths=0.3, alpha=0.4)
        for (px, py), nm in zip(pts, names):
            ax.plot(px, py, "o", mfc="w", mec="k", ms=7)
            ax.annotate(nm, (px, py), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.plot(xs[i], ys[j], "rX", ms=11, mec="k")
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax3 = fig.add_subplot(2, len(keys), len(keys) + n + 1, projection="3d")
        T = np.log10(Zp)
        ax3.plot_surface(X, Y, T, cmap=cm.GnBu_r, shade=False, linewidth=0, antialiased=True, rstride=1, cstride=1)
        for (px, py) in pts:
            ii, jj = np.argmin(np.abs(xs - px)), np.argmin(np.abs(ys - py))
            ax3.scatter([xs[ii]], [ys[jj]], [T[jj, ii]], color="w", edgecolors="k", s=35, depthshade=False)
        ax3.scatter([xs[i]], [ys[j]], [T[j, i]], color="r", marker="X", s=60, depthshade=False)
        ax3.set_axis_off(), ax3.set_box_aspect(None, zoom=1.1), ax3.view_init(elev=45, azim=235)
    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0, right=1, top=0.95, bottom=0)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2_r0.1_n41")
    ap.add_argument("--models", nargs="+", default=["sl_small", "ssl_small", "sl_high", "ssl_l1_high", "sl_fs_small"])
    ap.add_argument("--outdir", default="figures/landscape/paper")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    models = [m for m in args.models if os.path.exists(f"figures/landscape/random_{args.tag}_{m}.npz")]
    landscape_figure(args.tag, models, os.path.join(args.outdir, f"loss_vs_merit_{args.tag}.png"))
    training_curves(args.models, os.path.join(args.outdir, "training_curves.png"))
    for P in ["high", "fs", "l1", "fsmerit"]:
        for sfx in ["", "_test"]:  # _test: same weights and plane, losses on the held-out test split
            f = f"figures/landscape/plane_{P}_v2{sfx}_n41.npz"
            if os.path.exists(f):
                plane_figure(f, os.path.join(args.outdir, f"shared_plane_{P}{sfx}.png"))
    print(f"\nFigures in {args.outdir}")


if __name__ == "__main__":
    main()
