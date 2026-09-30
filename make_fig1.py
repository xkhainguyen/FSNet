"""Fig. 1: held-out training-loss landscapes of M1-M4 (all with the L1 penalty).

  M1 SL, small rho | M2 SL, large rho | M3 SL + FS, small rho | M4 SSL, small rho
Each model's own training loss, evaluated on held-out test samples, on a random
filter-normalized plane around its converged weights (run_fig1.sh). Every surface is
min-max normalized and drawn on the same log scale, so only the shape is compared.

  python make_fig1.py --radius 0.5
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from plot_landscape_compare import build_losses, ruggedness, normalize

OUT = "figures/landscape/fig1"
TITLES = {"M1": r"M1: SL, small $\rho$", "M2": r"M2: SL, large $\rho$",
          "M3": r"M3: SL + FS, small $\rho$", "M4": r"M4: SSL, small $\rho$"}
FLOOR = 1e-3  # log floor of the normalized loss


def own_loss(name, R, kind="random"):
    d = dict(np.load(f"{OUT}/{kind}_test_r{R}_{name}.npz", allow_pickle=True))
    L = build_losses(d, name, 10.0, 1e3, 1e5)
    c = {k: d[f"{name}/{k}"] for k in d["components"]}
    Z = {"M1": L["L_sl_l1_small"], "M2": L["L_sl_l1_high"],
         "M3": 100 * c["huber_fs"] + 10 * c["viol_l1"], "M4": L["L_ssl_l1_small"]}[name]
    return d["xs"], d["ys"], Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", default="0.5")
    ap.add_argument("--models", nargs="+", default=["M1", "M2", "M3", "M4"])
    ap.add_argument("--kind", choices=["random", "eig"], default="random",
                    help="eig: x axis = top Hessian eigenvector of the held-out loss")
    args = ap.parse_args()
    R, K = args.radius, args.kind
    models = [m for m in args.models if os.path.exists(f"{OUT}/{K}_test_r{R}_{m}.npz")]
    xlab = r"top Hessian direction" if K == "eig" else r"$\alpha$"

    data = {m: own_loss(m, R, K) for m in models}
    xs, ys = data[models[0]][0], data[models[0]][1]
    X, Y = np.meshgrid(xs, ys)
    jc, ic = len(ys) // 2, len(xs) // 2

    fig = plt.figure(figsize=(4.4 * len(models), 8.6))
    levels = np.geomspace(FLOOR, 1.0, 25)
    for n, m in enumerate(models):
        Zn = normalize(data[m][2])
        T = np.log10(Zn + FLOOR)
        j, i = np.unravel_index(np.argmin(Zn), Zn.shape)
        ax = fig.add_subplot(2, len(models), n + 1, projection="3d")
        ax.plot_surface(X, Y, T, cmap=cm.GnBu_r, vmin=np.log10(FLOOR), vmax=0, shade=False,
                        linewidth=0, antialiased=True, rstride=1, cstride=1)
        ax.scatter([0], [0], [T[jc, ic]], color="r", s=45, depthshade=False)
        ax.set_zlim(np.log10(FLOOR), 0)
        ax.set_axis_off(), ax.set_box_aspect(None, zoom=1.15), ax.view_init(elev=35, azim=190)
        ax.set_title(TITLES[m], fontsize=13, pad=-2)
        ax2 = fig.add_subplot(2, len(models), len(models) + n + 1)
        cs = ax2.contourf(X, Y, Zn + FLOOR, levels=levels, norm=colors.LogNorm(), cmap="GnBu_r", extend="both")
        ax2.contour(X, Y, Zn + FLOOR, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
        ax2.plot(0, 0, "o", mfc="r", mec="k", ms=8, label="trained weights")
        ax2.plot(xs[i], ys[j], "X", color="w", mec="k", ms=10, label="held-out minimum")
        ax2.set_aspect("equal"), ax2.set_xlabel(xlab)
        if n == 0:
            ax2.set_ylabel(r"$\beta$"), ax2.legend(fontsize=8, loc="lower left")
    fig.colorbar(cs, ax=fig.axes[1::2], shrink=0.6, label="normalized held-out training loss")
    fig.savefig(f"{OUT}/fig1_{K}_r{R}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for m in models:
        Zn = normalize(data[m][2])
        axes[0].plot(xs, Zn[jc, :] + FLOOR, label=TITLES[m])
        axes[1].plot(ys, Zn[:, ic] + FLOOR, label=TITLES[m])
    for ax, lab in zip(axes, [xlab + r" ($\beta = 0$)", r"$\beta$, random ($\alpha = 0$)"]):
        ax.set_yscale("log"), ax.set_xlabel(lab), ax.set_ylabel("normalized held-out loss")
    axes[0].legend(fontsize=8)
    fig.savefig(f"{OUT}/fig1_{K}_slices_r{R}.png", dpi=200)
    plt.close(fig)

    print(f"\n## Fig. 1 ({K} plane), radius {R}, held-out test split\n")
    print("| model | n local min | nonconvex frac | roughness | trained weights: normalized loss | held-out min at (a, b) | Hessian stiffness | Hessian kink |")
    print("|---|---:|---:|---:|---:|---|---:|---:|")
    for m in models:
        Z = data[m][2]
        r = ruggedness(Z, xs, ys)
        j, i = np.unravel_index(np.argmin(Z), Z.shape)
        h = {}
        if os.path.exists(f"{OUT}/hessian_test_{m}.json"):
            h = json.load(open(f"{OUT}/hessian_test_{m}.json"))["own"]
        print(f"| {m} | {r['n_min']} | {r['nonconvex']:.3f} | {r['roughness']:.2e} | {r['center']:.4f} | "
              f"({xs[i]:+.3f}, {ys[j]:+.3f}) | {h.get('stiffness', float('nan')):.3g} | {h.get('kink', float('nan')):.3g} |")
    print(f"\nFigures: {OUT}/fig1_{K}_r{R}.png, {OUT}/fig1_{K}_slices_r{R}.png")


if __name__ == "__main__":
    main()
