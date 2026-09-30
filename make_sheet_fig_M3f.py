"""Fig. 1 main candidate: 10-seed sheets (compute_sheet.py), M1 | M2 | M3f | M4, no titles.
z = log10(L / L_best_seed), capped at 3 decades, same scale and same 3D box for every panel.
Red: the 10 trained models."""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from fig1_common import own_loss

OUT = "figures/landscape/fig1"
MODELS = [("M1", r"SL, soft penalty, $\rho = 10^1$", "sheet_test_M1.npz"),
          ("M2", r"SL, soft penalty, $\rho = 10^5$", "sheet_test_M2.npz"),
          ("M3f", "SL, hard FS", "sheet_tri10_test_M3f.npz"),
          ("M4", r"SSL, soft penalty, $\rho = 10^1$", "sheet_test_M4.npz")]
CAP = 3.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="wm | act: use sheet10_{tag}_test_{M}.npz (run_actalign.sh)")
    tag = ap.parse_args().tag
    ncol = len(MODELS)
    fig = plt.figure(figsize=(6.5 * ncol, 11))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.1, 1], hspace=-0.05, wspace=0.02)
    seed_h = Line2D([], [], ls="", marker="o", mfc="r", mec="k", ms=9, label="trained models")
    ax2s = []
    for n, (m, label, fname) in enumerate(MODELS):
        f = f"{OUT}/{fname}" if tag is None else f"{OUT}/sheet10_{tag}_test_{m}.npz"
        if not os.path.exists(f):
            continue
        d = dict(np.load(f, allow_pickle=True))
        xs, ys, pts = d["xs"], d["ys"], d["points"]
        Z = own_loss(m, {k: d[f"plane/{k}"] for k in d["components"]})
        inside = np.isfinite(Z)
        if np.nanmin(Z) <= 0:
            Z = Z - np.nanmin(Z) + 1.0
        # each seed -> nearest grid point that lies on the sheet (corners can fall just outside)
        Jg, Ig = np.nonzero(inside)
        idx = [(int(Jg[k]), int(Ig[k])) for k in
               (np.argmin((xs[Ig] - p[0]) ** 2 + (ys[Jg] - p[1]) ** 2) for p in pts)]
        T = np.log10(Z / min(Z[k] for k in idx))
        Tc = np.where(inside, np.minimum(T, CAP), np.nan)
        X, Y = np.meshgrid(xs, ys)
        ax = fig.add_subplot(gs[0, n], projection="3d")
        ax.plot_surface(X, Y, Tc, cmap=cm.viridis, vmin=0, vmax=CAP, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, alpha=0.95)
        for (j, i) in idx:
            ax.scatter([xs[i]], [ys[j]], [Tc[j, i]], color="r", edgecolors="k", s=45, depthshade=False)
        ax.set_zlim(0, CAP), ax.set_zticks([0, 1, 2, 3]), ax.set_xticks([]), ax.set_yticks([])
        ax.view_init(elev=35, azim=-65)
        if n == 0:
            ax.set_zlabel(r"$\log_{10}(L / L_{\mathrm{best}})$", fontsize=13)
        ax2 = fig.add_subplot(gs[1, n])
        cs = ax2.contourf(X, Y, np.ma.masked_invalid(Tc), levels=np.linspace(0, CAP, 31), cmap="viridis")
        for (j, i) in idx:
            ax2.plot(xs[i], ys[j], "o", mfc="r", mec="k", ms=8)
        ax2.set_aspect("equal"), ax2.set_axis_off()
        ax2.legend(handles=[Line2D([], [], ls="", label=label)], loc="upper center", bbox_to_anchor=(0.5, 1.12),
                   handlelength=0, handletextpad=0, fontsize=16, frameon=True)
        ax2s.append(ax2)
    fig.legend(handles=[seed_h], loc="lower center", bbox_to_anchor=(0.45, 0.02), fontsize=14, frameon=False)
    fig.colorbar(cs, ax=ax2s, shrink=0.8, pad=0.01, ticks=[0, 1, 2, 3],
                 label=r"$\log_{10}(L / L_{\mathrm{best}})$, held-out")
    path = f"{OUT}/fig1_sheet10_M3f.png" if tag is None else f"{OUT}/fig1_sheet10_{tag}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
