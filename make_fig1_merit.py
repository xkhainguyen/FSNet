"""Fig. 1, old-paper style: merit after FS (obj + 1e5 * L1 violation of FS(y)) around each
model's converged weights, held-out test split, ONE shared absolute color scale (as in
visualize_weight_to_merit.ipynb), trained weights (red) vs grid minimum (white x)."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

OUT = "figures/landscape/fig1"
TITLES = {"M1": r"M1: SL, small $\rho$", "M2": r"M2: SL, large $\rho$",
          "M3": r"M3: SL + FS, small $\rho$", "M4": r"M4: SSL, small $\rho$"}
R = sys.argv[1] if len(sys.argv) > 1 else "0.5"
KIND = sys.argv[2] if len(sys.argv) > 2 else "random"
data = {}
for m in TITLES:
    d = dict(np.load(f"{OUT}/{KIND}_test_r{R}_{m}.npz", allow_pickle=True))
    data[m] = (d["xs"], d["ys"], d[f"{m}/obj_fs"] + 1e5 * d[f"{m}/viol_l1_fs"])
allZ = np.concatenate([z.ravel() for _, _, z in data.values()])
# shared absolute LOG scale; top just above the worst model's plateau so FS-failure walls saturate
vmin, vmax = max(allZ.min(), 1e-2), 1.5 * max(np.median(z) for _, _, z in data.values())
levels = np.geomspace(vmin, vmax, 25)
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharex=True, sharey=True, constrained_layout=True)
print(f"| model | merit after FS at weights | grid min | at (a, b) | weights / min |")
print("|---|---:|---:|---|---:|")
for ax, (m, (xs, ys, Z)) in zip(axes, data.items()):
    X, Y = np.meshgrid(xs, ys)
    cs = ax.contourf(X, Y, Z, levels=levels, norm=colors.LogNorm(vmin, vmax), cmap="viridis", extend="both")
    j, i = np.unravel_index(np.argmin(Z), Z.shape)
    c = Z[len(ys) // 2, len(xs) // 2]
    ax.plot(0, 0, "o", mfc="r", mec="k", ms=8)
    ax.plot(xs[i], ys[j], "X", color="w", mec="k", ms=11)
    ax.set_title(TITLES[m], fontsize=14), ax.set_xlabel(r"$\alpha$")
    print(f"| {m} | {c:.3g} | {Z.min():.3g} | ({xs[i]:+.2f}, {ys[j]:+.2f}) | {c / Z.min():.2f}x |")
axes[0].set_ylabel(r"$\beta$")
fig.colorbar(cs, ax=axes, label=r"merit $\mathcal{M}$ after FS (held-out)", shrink=0.9)
fig.savefig(f"{OUT}/fig1_merit_{KIND}_r{R}.png", dpi=200)
print(f"{OUT}/fig1_merit_{KIND}_r{R}.png")
