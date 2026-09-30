"""3D held-out SL loss of M1 (rho 10) and M2 (rho 1e5), L1 penalty, each on its own
(top Hessian eigenvector, random) plane. z = log10 of the min-max normalized loss.
Red: trained weights. Orange: 1D dips along the top direction (saddles in 2D)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from make_m1m2 import comps, sl_loss

OUT = "figures/landscape/fig1"
for R in ["0.1", "0.5"]:
    fig = plt.figure(figsize=(16, 7))
    for n, (m, rho, title) in enumerate([("M1", 10.0, r"M1: SL, small $\rho$"), ("M2", 1e5, r"M2: SL, large $\rho$")]):
        xs, ys, c = comps(m, R, "eig")
        Z = sl_loss(c, rho)
        T = np.log10((Z - Z.min()) / (Z.max() - Z.min()) + 1e-4)
        X, Y = np.meshgrid(xs, ys)
        ax = fig.add_subplot(1, 2, n + 1, projection="3d")
        ax.plot_surface(X, Y, T, cmap=cm.viridis, vmin=-4, vmax=0, rstride=1, cstride=1,
                        linewidth=0.1, edgecolor="k", alpha=0.95, antialiased=True)
        jc = len(ys) // 2
        ax.scatter([0], [0], [T[jc, jc]], color="r", s=80, depthshade=False, label="trained weights")
        line = Z[jc, :]
        dips = [i for i in range(1, len(xs) - 1) if i != jc and line[i] < line[i - 1] and line[i] < line[i + 1]]
        if dips:
            ax.scatter(xs[dips], ys[[jc] * len(dips)], T[jc, dips], color="orange", s=50, depthshade=False,
                       label="dip along top direction (saddle)")
        ax.set_xlabel("top Hessian direction"), ax.set_ylabel("random direction"), ax.set_zlabel("log10 normalized loss")
        ax.set_zlim(-4, 0), ax.view_init(elev=28, azim=-60)
        ax.set_title(title, fontsize=15)
        if n == 0:
            ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/m1_vs_m2_3d_r{R}.png", dpi=200)
    plt.close(fig)
    print(f"{OUT}/m1_vs_m2_3d_r{R}.png")
