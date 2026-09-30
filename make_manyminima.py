"""One landscape with many local minima: held-out plane through 3 seeds (run_wideplanes.sh),
own training loss, M1 vs M2 vs M4 (L1 penalty).

z = log10(L / L_best_seed), capped at CAP decades, same scale for every panel.
Red: the three trained seeds. White: every local minimum of the plane, i.e. a grid point
lower than all others in its (2k+1)x(2k+1) window (k = 2), not on the border.
M4 (SSL) loss can be negative, so it is shifted to a grid minimum of 1.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.lib.stride_tricks import sliding_window_view

OUT = "figures/landscape/fig1"
TAG = sys.argv[1] if len(sys.argv) > 1 else "wide"
MODELS = [("M1", r"M1: SL, small $\rho$"), ("M2", r"M2: SL, large $\rho$"), ("M4", r"M4: SSL, small $\rho$")]
CAP, K = 3.0, 2


def own(m, d):
    c = {k: d[f"plane/{k}"] for k in d["components"]}
    return {"M1": 100 * c["huber"] + 10 * c["viol_l1"], "M2": 100 * c["huber"] + 1e5 * c["viol_l1"],
            "M4": c["obj"] + 10 * c["viol_l1"]}[m]


def local_minima(Z, k=K):
    w = sliding_window_view(np.pad(Z, k, constant_values=np.inf), (2 * k + 1, 2 * k + 1))
    center = Z
    others = w.reshape(*Z.shape, -1)
    mid = (2 * k + 1) ** 2 // 2
    others = np.delete(others, mid, axis=-1)
    is_min = (center[..., None] < others).all(-1)
    is_min[:k, :] = is_min[-k:, :] = is_min[:, :k] = is_min[:, -k:] = False
    return np.nonzero(is_min)


def main():
    fig = plt.figure(figsize=(21, 13))
    print("| model | local minima in the plane | within 1 decade of best seed | depth of minima (decades above best seed) |")
    print("|---|---:|---:|---|")
    n = 0
    for m, title in MODELS:
        f = f"{OUT}/seedplane_{TAG}_test_{m}.npz"
        if not os.path.exists(f):
            continue
        n += 1
        d = dict(np.load(f, allow_pickle=True))
        xs, ys, pts = d["xs"], d["ys"], d["points"]
        Z = own(m, d)
        if Z.min() <= 0:
            Z = Z - Z.min() + 1.0
        idx = [(int(np.argmin(np.abs(ys - p[1]))), int(np.argmin(np.abs(xs - p[0])))) for p in pts]
        zbest = min(Z[k] for k in idx)
        T = np.log10(Z / zbest)
        jm, im = local_minima(Z)
        depth = T[jm, im]
        print(f"| {m} | {len(jm)} | {(depth < 1).sum()} | " + ", ".join(f"{v:.2f}" for v in np.sort(depth)[:12]) + " |")
        Tc = np.minimum(T, CAP)
        X, Y = np.meshgrid(xs, ys)
        ax = fig.add_subplot(2, 3, n, projection="3d")
        ax.plot_surface(X, Y, Tc, cmap=cm.viridis, vmin=0, vmax=CAP, rstride=2, cstride=2,
                        linewidth=0.02, edgecolor="k", alpha=0.9)
        ax.scatter(xs[im], ys[jm], Tc[jm, im], color="w", edgecolors="k", s=28, depthshade=False, label="local minima")
        for (j, i) in idx:
            ax.scatter([xs[i]], [ys[j]], [Tc[j, i]], color="r", s=60, depthshade=False)
        ax.set_zlim(-0.1, CAP), ax.view_init(elev=42, azim=-65)
        ax.set_title(f"{title}\n{len(jm)} local minima in this slice", fontsize=13)
        if n == 1:
            ax.set_zlabel(r"$\log_{10}(L / L_{\mathrm{best\ seed}})$"), ax.legend(loc="upper left", fontsize=9)
        ax2 = fig.add_subplot(2, 3, 3 + n)
        cs = ax2.contourf(X, Y, Tc, levels=np.linspace(0, CAP, 31), cmap="viridis")
        ax2.plot(xs[im], ys[jm], "o", mfc="w", mec="k", ms=5)
        for (j, i), lab in zip(idx, [str(x) for x in d["names"]]):
            ax2.plot(xs[i], ys[j], "o", mfc="r", mec="k", ms=9)
            ax2.annotate(lab, (xs[i], ys[j]), textcoords="offset points", xytext=(5, 5), color="w", fontsize=11)
        ax2.set_aspect("equal")
    fig.colorbar(cs, ax=fig.axes[1::2], shrink=0.8, label=r"$\log_{10}(L / L_{\mathrm{best\ seed}})$ (shared)")
    fig.savefig(f"{OUT}/fig1_manyminima_{TAG}.png", dpi=180, bbox_inches="tight")
    print(f"{OUT}/fig1_manyminima_{TAG}.png")


if __name__ == "__main__":
    main()
