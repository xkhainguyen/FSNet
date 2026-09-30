"""fig1_manyminima_aligned.png style for the 4 methods of fig1_sheet10_M3f.png (M1, M2, M3f, M4).

Held-out wide plane through permutation-aligned seeds 0, 1, 2 (run_alignedplanes.sh, and
run_alignedplane_M3f.sh with the FS layer), own training loss (fig1_common.own_loss).
z = log10(L / L_best_seed), capped at CAP decades, same scale for every panel.
Red: the three trained seeds. White: local minima of the plane (make_manyminima.local_minima).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from fig1_common import own_loss
from make_manyminima import local_minima

OUT = "figures/landscape/fig1"
MODELS = [("M1", r"SL, soft penalty, $\rho = 10^1$"), ("M2", r"SL, soft penalty, $\rho = 10^5$"),
          ("M3f", "SL, hard FS"), ("M4", r"SSL, soft penalty, $\rho = 10^1$")]
CAP = 3.0


def main():
    ncol = len(MODELS)
    fig = plt.figure(figsize=(6.5 * ncol, 12))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.1, 1], hspace=0.08, wspace=0.08)
    ax2s = []
    print("| method | local minima in the plane | within 1 decade of best seed |")
    print("|---|---:|---:|")
    for n, (m, label) in enumerate(MODELS):
        f = f"{OUT}/seedplane_aligned_test_{m}.npz"
        if not os.path.exists(f):
            print(f"missing {f}")
            continue
        d = dict(np.load(f, allow_pickle=True))
        xs, ys, pts = d["xs"], d["ys"], d["points"]
        Z = own_loss(m, {k: d[f"plane/{k}"] for k in d["components"]})
        if Z.min() <= 0:
            Z = Z - Z.min() + 1.0
        idx = [(int(np.argmin(np.abs(ys - p[1]))), int(np.argmin(np.abs(xs - p[0])))) for p in pts]
        T = np.log10(Z / min(Z[k] for k in idx))
        jm, im = local_minima(Z)
        print(f"| {m} | {len(jm)} | {(T[jm, im] < 1).sum()} |")
        Tc = np.clip(T, 0, CAP)
        X, Y = np.meshgrid(xs, ys)
        ax = fig.add_subplot(gs[0, n], projection="3d")
        ax.plot_surface(X, Y, Tc, cmap=cm.viridis, vmin=0, vmax=CAP, rstride=2, cstride=2,
                        linewidth=0.02, edgecolor="k", alpha=0.9)
        ax.scatter(xs[im], ys[jm], Tc[jm, im], color="w", edgecolors="k", s=28, depthshade=False)
        for (j, i) in idx:
            ax.scatter([xs[i]], [ys[j]], [Tc[j, i]], color="r", edgecolors="k", s=60, depthshade=False)
        ax.set_zlim(0, CAP), ax.set_zticks([0, 1, 2, 3]), ax.view_init(elev=42, azim=-65)
        if n == 0:
            ax.set_zlabel(r"$\log_{10}(L / L_{\mathrm{best}})$", fontsize=13)
        ax2 = fig.add_subplot(gs[1, n])
        cs = ax2.contourf(X, Y, Tc, levels=np.linspace(0, CAP, 31), cmap="viridis")
        ax2.plot(xs[im], ys[jm], "o", mfc="w", mec="k", ms=5)
        for (j, i) in idx:
            ax2.plot(xs[i], ys[j], "o", mfc="r", mec="k", ms=9)
        ax2.set_aspect("equal"), ax2.set_xticks([]), ax2.set_yticks([])
        ax2.legend(handles=[Line2D([], [], ls="", label=label)], loc="upper center", bbox_to_anchor=(0.5, 1.13),
                   handlelength=0, handletextpad=0, fontsize=16, frameon=True)
        ax2s.append(ax2)
    keys = [Line2D([], [], ls="", marker="o", mfc="r", mec="k", ms=10, label="trained models"),
            Line2D([], [], ls="", marker="o", mfc="w", mec="k", ms=7, label="local minima")]
    fig.legend(handles=keys, loc="lower center", ncol=2, bbox_to_anchor=(0.45, 0.04), fontsize=14, frameon=False)
    fig.colorbar(cs, ax=ax2s, shrink=0.8, pad=0.01, ticks=[0, 1, 2, 3],
                 label=r"$\log_{10}(L / L_{\mathrm{best}})$, held-out")
    path = f"{OUT}/fig1_manyminima4.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
