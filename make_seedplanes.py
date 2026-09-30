"""Multiple local minima of M1 vs M2 (SL, L1 penalty, rho 10 vs 1e5), held-out test split.

1. Plane through three independently trained seeds of each model (run_seedplanes.sh):
   3D + contour of the own SL loss, seeds marked; strict grid local minima; and the
   barrier on the straight path between each pair of seeds,
   barrier = max on the path / max(loss at the two ends). > 1 means the two seeds sit in
   separate basins; ~1 means a connected (monotone) path.
2. Random plane zoomed to +-0.01 around seed 0 (fine-scale structure).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.interpolate import RegularGridInterpolator

OUT = "figures/landscape/fig1"
MODELS = [("M1", 10.0, r"M1: SL, small $\rho$ (10)"), ("M2", 1e5, r"M2: SL, large $\rho$ (1e5)")]


def sl(d, prefix, rho):
    return 100 * d[f"{prefix}/huber"] + rho * d[f"{prefix}/viol_l1"]


def local_minima(Z):
    inner = Z[1:-1, 1:-1]
    nb = np.stack([Z[1 + a:Z.shape[0] - 1 + a, 1 + b:Z.shape[1] - 1 + b]
                   for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)])
    j, i = np.nonzero((inner[None] < nb).all(0))
    return j + 1, i + 1


def main():
    fig = plt.figure(figsize=(16, 13))
    print("| model | grid local minima | barrier s0-s1 | barrier s0-s2 | barrier s1-s2 | loss at seeds |")
    print("|---|---:|---:|---:|---:|---|")
    for n, (m, rho, title) in enumerate(MODELS):
        f = f"{OUT}/seedplane_test_{m}.npz"
        if not os.path.exists(f):
            continue
        d = dict(np.load(f, allow_pickle=True))
        xs, ys, pts = d["xs"], d["ys"], d["points"]
        Z = sl(d, "plane", rho)
        interp = RegularGridInterpolator((ys, xs), np.log(Z))
        at = [float(np.exp(interp([[p[1], p[0]]])[0])) for p in pts]
        bars = []
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            t = np.linspace(0, 1, 201)[:, None]
            path = pts[a] + t * (pts[b] - pts[a])
            vals = np.exp(interp(path[:, ::-1]))
            bars.append(vals.max() / max(at[a], at[b]))
        jm, im = local_minima(Z)
        print(f"| {m} | {len(jm)} | {bars[0]:.2f} | {bars[1]:.2f} | {bars[2]:.2f} | "
              + ", ".join(f"{v:.4g}" for v in at) + " |")

        X, Y = np.meshgrid(xs, ys)
        T = np.log10(Z)
        ax = fig.add_subplot(3, 2, n + 1, projection="3d")
        ax.plot_surface(X, Y, T, cmap=cm.viridis, rstride=1, cstride=1, linewidth=0.1, edgecolor="k", alpha=0.95)
        for p, lab in zip(pts, ["s0", "s1", "s2"]):
            ii, jj = np.argmin(np.abs(xs - p[0])), np.argmin(np.abs(ys - p[1]))
            ax.scatter([xs[ii]], [ys[jj]], [T[jj, ii]], color="r", s=70, depthshade=False)
            ax.text(xs[ii], ys[jj], T[jj, ii], "  " + lab, color="r", fontsize=11)
        ax.set_title(title + ": plane through 3 seeds", fontsize=13)
        ax.set_zlabel("log10 held-out training loss"), ax.view_init(elev=35, azim=-60)

        ax2 = fig.add_subplot(3, 2, n + 3)
        lv = np.linspace(T.min(), np.percentile(T, 98), 30)
        ax2.contourf(X, Y, T, levels=lv, cmap="viridis", extend="max")
        ax2.contour(X, Y, T, levels=lv, colors="k", linewidths=0.3, alpha=0.5)
        ax2.plot(xs[im], ys[jm], "wx", ms=8, mew=2, label="grid local minima")
        for p, lab in zip(pts, ["s0", "s1", "s2"]):
            ax2.plot(*p, "o", mfc="r", mec="k", ms=9)
            ax2.annotate(lab, p, textcoords="offset points", xytext=(6, 6), color="w", fontsize=11)
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            ax2.plot(*zip(pts[a], pts[b]), "w--", lw=0.8)
        ax2.set_aspect("equal"), ax2.legend(fontsize=8, loc="lower right")

        fz = f"{OUT}/random_test_r0.01_{m}.npz"
        if os.path.exists(fz):
            dz = dict(np.load(fz, allow_pickle=True))
            Zz = sl(dz, m, rho)
            Xz, Yz = np.meshgrid(dz["xs"], dz["ys"])
            ax3 = fig.add_subplot(3, 2, n + 5, projection="3d")
            ax3.plot_surface(Xz, Yz, Zz, cmap=cm.viridis, rstride=1, cstride=1, linewidth=0.1, edgecolor="k", alpha=0.95)
            jc = len(dz["ys"]) // 2
            ax3.scatter([0], [0], [Zz[jc, jc]], color="r", s=70, depthshade=False)
            ax3.set_title(title + r": random plane, zoom $\pm$0.01", fontsize=12)
            ax3.set_zlabel("held-out training loss"), ax3.view_init(elev=30, azim=-60)
            jz, _ = local_minima(Zz)
            print(f"  {m} zoom +-0.01: grid local minima {len(jz)}, loss range {Zz.min():.5g} .. {Zz.max():.5g}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/m1_vs_m2_seedplanes.png", dpi=200)
    print(f"{OUT}/m1_vs_m2_seedplanes.png")


if __name__ == "__main__":
    main()
