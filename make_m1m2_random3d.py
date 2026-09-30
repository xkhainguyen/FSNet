"""M1 vs M2: own training loss (100 huber + rho * L1 violation) over two random
filter-normalized directions around the converged weights, held-out test split.
z = log10(loss), real values, one axis per model."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from make_m1m2 import comps, sl_loss

OUT = "figures/landscape/fig1"
for R in sys.argv[1:] or ["0.1", "0.5", "1.0"]:
    if not all(os.path.exists(f"{OUT}/random_test_r{R}_{m}.npz") for m in ("M1", "M2")):
        continue
    fig = plt.figure(figsize=(16, 7))
    for n, (m, rho, title) in enumerate([("M1", 10.0, r"M1: SL, small $\rho$ (10)"), ("M2", 1e5, r"M2: SL, large $\rho$ (1e5)")]):
        xs, ys, c = comps(m, R, "random")
        T = np.log10(sl_loss(c, rho))
        X, Y = np.meshgrid(xs, ys)
        ax = fig.add_subplot(1, 2, n + 1, projection="3d")
        ax.plot_surface(X, Y, T, cmap=cm.viridis, rstride=1, cstride=1, linewidth=0.1, edgecolor="k", alpha=0.95)
        jc = len(ys) // 2
        ax.scatter([0], [0], [T[jc, jc]], color="r", s=80, depthshade=False, label="trained weights")
        ax.set_xlabel(r"random direction $\alpha$"), ax.set_ylabel(r"random direction $\beta$")
        ax.set_zlabel("log10 held-out training loss"), ax.view_init(elev=28, azim=-60)
        ax.set_title(title, fontsize=15)
        if n == 0:
            ax.legend(loc="upper left", fontsize=9)
        print(f"r={R} {m}: loss at weights {10 ** T[jc, jc]:.4g}, edge mean {10 ** np.concatenate([T[0], T[-1]]).mean():.4g}, "
              f"max {10 ** T.max():.4g} ({T.max() - T[jc, jc]:.2f} decades above)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/m1_vs_m2_random3d_r{R}.png", dpi=200)
    plt.close(fig)
    print(f"{OUT}/m1_vs_m2_random3d_r{R}.png")
