"""3D held-out training-loss landscapes of M1-M4 (L1 penalty) over two random
filter-normalized directions, radius 1, for seeds 0-2 (run_random3d_seeds.sh).
z = log10(own training loss), one z axis per panel (the losses have different scales).

Printed per panel: decades of rise from the weights to the ring at radius 0.1 and to the
edge (radius 1), and the inner share = rise within 0.1 / total rise
(large = needle-like minimum, small = broad bowl).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

OUT = "figures/landscape/fig1"
MODELS = [("M1", r"M1: SL, small $\rho$"), ("M2", r"M2: SL, large $\rho$"),
          ("M3", r"M3: SL + FS, small $\rho$"), ("M4", r"M4: SSL, small $\rho$")]
SEEDS = [0, 1, 2]


def own(m, d):
    c = {k: d[f"{m}/{k}"] for k in d["components"]}
    return {"M1": 100 * c["huber"] + 10 * c["viol_l1"], "M2": 100 * c["huber"] + 1e5 * c["viol_l1"],
            "M3": 100 * c["huber_fs"] + 10 * c["viol_l1"], "M4": c["obj"] + 10 * c["viol_l1"]}[m]


def ring(Z, xs, r, tol):
    X, Y = np.meshgrid(xs, xs)
    R = np.maximum(np.abs(X), np.abs(Y))
    return Z[np.abs(R - r) <= tol].mean()


def main():
    fig = plt.figure(figsize=(20, 15))
    print("| model | seed | loss at weights | rise to 0.1 (decades) | rise to edge (decades) | inner share |")
    print("|---|---:|---:|---:|---:|---:|")
    for r_, s in enumerate(SEEDS):
        for c_, (m, title) in enumerate(MODELS):
            f = f"{OUT}/random_test_r1.0_{m}{'' if s == 0 else f'_s{s}'}.npz"
            if not os.path.exists(f):
                continue
            d = dict(np.load(f, allow_pickle=True))
            xs, ys = d["xs"], d["ys"]
            Z = own(m, d)
            if m == "M4" and Z.min() <= 0:
                Z = Z - Z.min() + 1e-3 * (Z.max() - Z.min())  # SSL objective can be negative
            T = np.log10(Z)
            jc = len(ys) // 2
            h = xs[1] - xs[0]
            inner = np.log10(ring(Z, xs, 0.1, h / 2) / Z[jc, jc])
            total = np.log10(ring(Z, xs, 1.0, h / 2) / Z[jc, jc])
            print(f"| {m} | {s} | {Z[jc, jc]:.4g} | {inner:.2f} | {total:.2f} | {inner / total:.2f} |")
            X, Y = np.meshgrid(xs, ys)
            ax = fig.add_subplot(len(SEEDS), len(MODELS), r_ * len(MODELS) + c_ + 1, projection="3d")
            ax.plot_surface(X, Y, T, cmap=cm.viridis, rstride=1, cstride=1, linewidth=0.05, edgecolor="k", alpha=0.95)
            ax.scatter([0], [0], [T[jc, jc]], color="r", s=50, depthshade=False)
            ax.view_init(elev=28, azim=-60)
            ax.set_xticks([-1, 0, 1]), ax.set_yticks([-1, 0, 1])
            ax.set_title(f"{title}, seed {s}", fontsize=12)
            if c_ == 0:
                ax.set_zlabel("log10 held-out loss")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_random3d_seeds.png", dpi=170)
    print(f"{OUT}/fig1_random3d_seeds.png")


if __name__ == "__main__":
    main()
