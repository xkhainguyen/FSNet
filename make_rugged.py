"""Ruggedness of the held-out training loss on the permutation-aligned seed planes (M1, M2, M4).

1. Basins of attraction: every grid point follows steepest descent on the 8-connected grid
   (move to the lowest neighbour while it is lower) until it reaches a local minimum; points
   are coloured by the minimum they end in. Many small basins = rugged; few large = smooth.
2. Curvature type of log L from the 2x2 finite-difference Hessian: convex (both eigenvalues > 0),
   saddle (opposite signs), concave (both < 0). Eigenvalues below 1% of the plane's median
   |eigenvalue| count as flat.
Printed: number of basins, share of the plane in the largest basin, saddle / concave fraction.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from make_manyminima import own

OUT = "figures/landscape/fig1"
MODELS = [("M1", r"M1: SL, small $\rho$"), ("M2", r"M2: SL, large $\rho$"), ("M4", r"M4: SSL, small $\rho$")]
TAGS = ["aligned", "aligned_t345", "aligned_t678"]


def basins(Z):
    ny, nx = Z.shape
    P = np.pad(Z, 1, constant_values=np.inf)
    nb = np.stack([P[1 + a:ny + 1 + a, 1 + b:nx + 1 + b] for a in (-1, 0, 1) for b in (-1, 0, 1)])
    offs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    k = nb.argmin(0)  # index 4 is the point itself: a local minimum
    nxt_j = np.arange(ny)[:, None] + np.array([o[0] for o in offs])[k]
    nxt_i = np.arange(nx)[None, :] + np.array([o[1] for o in offs])[k]
    J, I = np.indices(Z.shape)
    for _ in range(ny * nx):
        nj, ni = nxt_j[J, I], nxt_i[J, I]
        if np.array_equal(nj, J) and np.array_equal(ni, I):
            break
        J, I = nj, ni
    lab = J * nx + I
    _, lab = np.unique(lab, return_inverse=True)
    return lab.reshape(Z.shape)


def curvature_type(T, h):
    zxx = (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / h ** 2
    zyy = (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / h ** 2
    zxy = (T[2:, 2:] - T[2:, :-2] - T[:-2, 2:] + T[:-2, :-2]) / (4 * h ** 2)
    tr, det = zxx + zyy, zxx * zyy - zxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    eps = 0.01 * np.median(np.abs(np.concatenate([l1.ravel(), l2.ravel()])))
    c = np.zeros(l1.shape, int)  # 0 flat, 1 convex, 2 saddle, 3 concave
    c[(l1 > eps) & (l2 > eps)] = 1
    c[(l1 > eps) & (l2 < -eps)] = 2
    c[(l1 < -eps) & (l2 < -eps)] = 3
    return c


def main():
    print("| model | slice | basins of attraction | largest basin (share of plane) | saddle | concave |")
    print("|---|---|---:|---:|---:|---:|")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    for n, (m, title) in enumerate(MODELS):
        for t, tag in enumerate(TAGS):
            d = dict(np.load(f"{OUT}/seedplane_{tag}_test_{m}.npz", allow_pickle=True))
            xs, ys, pts = d["xs"], d["ys"], d["points"]
            Z = own(m, d)
            if Z.min() <= 0:
                Z = Z - Z.min() + 1.0
            T = np.log10(Z)
            lab = basins(Z)
            sizes = np.bincount(lab.ravel()) / lab.size
            ct = curvature_type(T, xs[1] - xs[0])
            print(f"| {m} | {tag} | {len(sizes)} | {sizes.max():.0%} | {(ct == 2).mean():.0%} | {(ct == 3).mean():.0%} |")
            if t:
                continue
            X, Y = np.meshgrid(xs, ys)
            rng = np.random.default_rng(1)
            order = rng.permutation(len(sizes))
            cmap = plt.get_cmap("tab20", 20)
            ax = axes[0, n]
            ax.pcolormesh(X, Y, order[lab] % 20, cmap=cmap, shading="auto")
            ax.contour(X, Y, T, levels=15, colors="k", linewidths=0.3, alpha=0.4)
            for p, name in zip(pts, d["names"]):
                ax.plot(*p, "o", mfc="r", mec="k", ms=8)
                ax.annotate(str(name), p, textcoords="offset points", xytext=(5, 5), fontsize=10)
            ax.set_title(f"{title}\n{len(sizes)} basins of attraction, largest {sizes.max():.0%}", fontsize=12)
            ax.set_aspect("equal")
            ax = axes[1, n]
            cm4 = colors.ListedColormap(["#dddddd", "#2c7bb6", "#fdae61", "#d7191c"])
            ax.pcolormesh(X[1:-1, 1:-1], Y[1:-1, 1:-1], ct, cmap=cm4, vmin=-0.5, vmax=3.5, shading="auto")
            for p in pts:
                ax.plot(*p, "o", mfc="w", mec="k", ms=8)
            ax.set_title(f"curvature: saddle {(ct == 2).mean():.0%}, concave {(ct == 3).mean():.0%}", fontsize=12)
            ax.set_aspect("equal")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#dddddd", "#2c7bb6", "#fdae61", "#d7191c"]]
    axes[1, 2].legend(handles, ["flat", "convex", "saddle", "concave"], loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_rugged.png", dpi=180)
    print(f"{OUT}/fig1_rugged.png")


if __name__ == "__main__":
    main()
