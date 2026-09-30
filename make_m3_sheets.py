"""3-seed sheets (aligned seeds 0-2, one triangle) for M1 vs M3 at rho 10 / 1 / 0.8, own training loss
(M3: label term after FS), held-out, one shared log scale (decades above the best seed)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from fig1_common import TITLE, own_loss

OUT, CAP = "figures/landscape/fig1", 3.0
MODELS = ["M1", "M3", "M3r1", "M3r08"]
fig = plt.figure(figsize=(5.5 * len(MODELS), 10))
cs = None
for n, m in enumerate(MODELS):
    f = f"{OUT}/sheet_tri3_test_{m}.npz"
    if not os.path.exists(f):
        continue
    d = dict(np.load(f, allow_pickle=True))
    xs, ys, pts = d["xs"], d["ys"], d["points"]
    Z = own_loss(m, {k: d[f"plane/{k}"] for k in d["components"]})
    inside = np.isfinite(Z)
    Jg, Ig = np.nonzero(inside)
    idx = [(int(Jg[k]), int(Ig[k])) for k in (np.argmin((xs[Ig] - p[0]) ** 2 + (ys[Jg] - p[1]) ** 2) for p in pts)]
    T = np.where(inside, np.minimum(np.log10(Z / min(Z[k] for k in idx)), CAP), np.nan)
    X, Y = np.meshgrid(xs, ys)
    ax = fig.add_subplot(2, len(MODELS), n + 1)
    cs = ax.contourf(X, Y, np.ma.masked_invalid(T), levels=np.linspace(0, CAP, 31), cmap="viridis")
    for j, i in idx:
        ax.plot(xs[i], ys[j], "o", mfc="r", mec="k", ms=8)
    ax.set_title(f"{TITLE[m]}\nmax on sheet {10 ** np.nanmax(T):.1f}x the best seed", fontsize=12)
    ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    ax3 = fig.add_subplot(2, len(MODELS), len(MODELS) + n + 1, projection="3d")
    ax3.plot_surface(X, Y, T, cmap=cm.viridis, vmin=0, vmax=CAP, rstride=1, cstride=1, linewidth=0, alpha=0.95)
    ax3.set_zlim(0, CAP), ax3.set_axis_off(), ax3.view_init(elev=45, azim=-65)
fig.colorbar(cs, ax=fig.axes, shrink=0.6, label=r"$\log_{10}(L / L_{\mathrm{best\ seed}})$, held-out")
fig.savefig(f"{OUT}/fig1_m3_rho_sheets.png", dpi=180, bbox_inches="tight")
print(f"{OUT}/fig1_m3_rho_sheets.png")
