"""Same style as fig1_manyminima_aligned.png, on the 10-seed sheet (compute_sheet.py).
z = log10(L / L_best_seed), capped at 3 decades, shared scale. Red: the 10 trained seeds.
White: local minima of the sheet (lowest in a 5x5 window, not on the boundary)."""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from make_manyminima import local_minima
from fig1_common import own_loss

OUT = "figures/landscape/fig1"
MODELS = [("M1", r"M1: SL, small $\rho$", "sheet_test_M1.npz"), ("M2", r"M2: SL, large $\rho$", "sheet_test_M2.npz"),
          ("M3r1", r"M3: SL + FS, $\rho = 1$", "sheet_tri10_test_M3r1.npz"), ("M3r08", r"M3: SL + FS, $\rho = 0.8$", "sheet_tri10_test_M3r08.npz"),
          ("M3f", r"M3f: SL + FS, FSNet-style", "sheet_tri10_test_M3f.npz"), ("M4", r"M4: SSL, small $\rho$", "sheet_test_M4.npz")]
CAP = 3.0


def main():
    fig = plt.figure(figsize=(41, 12.5))
    ncol = len(MODELS)
    print("| model | local minima on the sheet | deep (< 1 decade above best seed) | seeds that are local minima |")
    print("|---|---:|---:|---:|")
    n = 0
    for m, title, fname in MODELS:
        f = f"{OUT}/{fname}"
        if not os.path.exists(f):
            continue
        n += 1
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
        zbest = min(Z[k] for k in idx if np.isfinite(Z[k]))
        T = np.log10(Z / zbest)
        Zs = np.where(inside, Z, np.inf)
        jm, im = local_minima(Zs)
        # points outside the sheet are +inf, so a boundary point is a minimum if it is lower than
        # every neighbour that lies on the sheet
        deep = (T[jm, im] < 1).sum()
        seed_mins = sum(any(abs(j - a) <= 2 and abs(i - b) <= 2 for a, b in zip(jm, im)) for j, i in idx)
        print(f"| {m} | {len(jm)} | {deep} | {seed_mins} / {len(pts)} |")
        Tc = np.where(inside, np.minimum(T, CAP), np.nan)
        X, Y = np.meshgrid(xs, ys)
        ax = fig.add_subplot(2, ncol, n, projection="3d")
        ax.plot_surface(X, Y, Tc, cmap=cm.viridis, vmin=0, vmax=CAP, rstride=1, cstride=1,
                        linewidth=0.02, edgecolor="k", alpha=0.92)
        ax.scatter(xs[im], ys[jm], Tc[jm, im], color="w", edgecolors="k", s=26, depthshade=False, label="local minima")
        for (j, i) in idx:
            ax.scatter([xs[i]], [ys[j]], [Tc[j, i]], color="r", s=50, depthshade=False)
        ax.set_zlim(-0.1, CAP), ax.view_init(elev=45, azim=-65)
        ns = len(pts)
        ax.set_title(f"{title}\n{len(jm)} local minima on a sheet through {ns} seeds ({deep} deep)"
                     + ("\n(only 3 seeds trained so far)" if ns == 3 else ""), fontsize=13)
        if n == 1:
            ax.set_zlabel(r"$\log_{10}(L / L_{\mathrm{best\ seed}})$"), ax.legend(loc="upper left", fontsize=9)
        ax2 = fig.add_subplot(2, ncol, ncol + n)
        cs = ax2.contourf(X, Y, np.ma.masked_invalid(Tc), levels=np.linspace(0, CAP, 31), cmap="viridis")
        ax2.plot(xs[im], ys[jm], "o", mfc="w", mec="k", ms=5)
        for (j, i), lab in zip(idx, d["names"]):
            ax2.plot(xs[i], ys[j], "o", mfc="r", mec="k", ms=8)
            ax2.annotate(str(lab), (xs[i], ys[j]), textcoords="offset points", xytext=(4, 4), color="w", fontsize=9)
        ax2.set_aspect("equal"), ax2.set_xticks([]), ax2.set_yticks([])
    fig.colorbar(cs, ax=fig.axes[1::2], shrink=0.8, label=r"$\log_{10}(L / L_{\mathrm{best\ seed}})$ (shared)")
    fig.savefig(f"{OUT}/fig1_sheet10_with_rho08.png", dpi=180, bbox_inches="tight")
    print(f"{OUT}/fig1_sheet10_with_rho08.png")


if __name__ == "__main__":
    main()
