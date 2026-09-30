"""Final Fig. 1: 2D panels primary, 3D below; hex19 seed sheets (compute_sheet.py).

Main:          M1, M2, M4, held-out split, ordering 0; titles give barrier statistics pooled over
               orderings 0-2 (sheet_stats.py).
Supplementary: M2 at M1's lr (M2m), and all four models on the training split (ordering 0).
Writes paper_figs/fig1/fig1_final.png, fig1_final_supp.png, fig1_final_caption.md.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from fig1_common import FIG, TITLE, own_loss
from sheet_stats import stats

CAP = 3.0
OUT = "paper_figs/fig1"


def sheet(m, order=0, split="test"):
    f = f"{FIG}/sheet_hex19_o{order}_{split}_{m}.npz"
    if not os.path.exists(f):
        return None
    d = dict(np.load(f, allow_pickle=True))
    c = {k: d[f"plane/{k}"] for k in d["components"]}
    Z = own_loss(m, c)
    inside = np.isfinite(Z)
    if np.nanmin(Z) <= 0:
        Z = Z - np.nanmin(Z) + 1.0
    xs, ys = d["xs"], d["ys"]
    Jg, Ig = np.nonzero(inside)  # nearest grid point on the sheet (corner seeds can fall just outside)
    idx = [(int(Jg[k]), int(Ig[k])) for k in (np.argmin((xs[Ig] - p[0]) ** 2 + (ys[Jg] - p[1]) ** 2) for p in d["points"])]
    T = np.log10(Z / min(Z[k] for k in idx))
    return d, np.where(inside, np.minimum(T, CAP), np.nan), idx


def pooled(m, split="test"):
    P, E = [], []
    for o in range(3):
        f = f"{FIG}/sheet_hex19_o{o}_{split}_{m}.npz"
        if os.path.exists(f):
            p, e, _ = stats(f, m)
            P += p
            E += e
    return np.median(P) if P else np.nan, np.median(E) if E else np.nan, len(P), len(E)


def panel_row(fig, gs_row, models, split, label_stats=True):
    cs = None
    for n, m in enumerate(models):
        s = sheet(m, 0, split)
        if s is None:
            continue
        d, Tc, idx = s
        X, Y = np.meshgrid(d["xs"], d["ys"])
        ax = fig.add_subplot(gs_row[0, n])
        cs = ax.contourf(X, Y, np.ma.masked_invalid(Tc), levels=np.linspace(0, CAP, 31), cmap="viridis")
        for (j, i) in idx:
            ax.plot(d["xs"][i], d["ys"][j], "o", mfc="r", mec="k", ms=6)
        pb, eb, npb, neb = pooled(m, split)
        title = TITLE[m]
        if label_stats:
            title += f"\nbarrier: pair {10 ** pb:.0f}x, escape {10 ** eb:.0f}x"
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax3 = fig.add_subplot(gs_row[1, n], projection="3d")
        ax3.plot_surface(X, Y, Tc, cmap=cm.viridis, vmin=0, vmax=CAP, rstride=2, cstride=2, linewidth=0, alpha=0.95)
        ax3.set_zlim(0, CAP), ax3.set_axis_off(), ax3.view_init(elev=50, azim=-60)
    return cs


def figure(models, split, path):
    fig = plt.figure(figsize=(6.0 * len(models), 9.5))
    gs = fig.add_gridspec(2, len(models), height_ratios=[1.25, 1], hspace=0.05, wspace=0.05)
    cs = panel_row(fig, gs, models, split)
    if cs is not None:
        fig.colorbar(cs, ax=fig.axes, shrink=0.6, pad=0.02,
                     label=r"$\log_{10}(L / L_{\mathrm{best\ seed}})$, held-out" if split == "test"
                     else r"$\log_{10}(L / L_{\mathrm{best\ seed}})$, training split")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    figure(["M1", "M2", "M4"], "test", f"{OUT}/fig1_final.png")
    figure(["M2m"], "test", f"{OUT}/fig1_final_supp_M2m.png")
    figure(["M1", "M2", "M2m", "M4"], "train", f"{OUT}/fig1_final_supp_train.png")
    rows = []
    for m in ["M1", "M2", "M2m", "M4"]:
        for split in ["test", "train"]:
            pb, eb, npb, neb = pooled(m, split)
            rows.append(f"| {m} | {split} | {10 ** pb:.1f}x ({npb} pairs) | {10 ** eb:.1f}x ({neb} seeds) |")
    caption = f"""# Fig. 1 caption draft

Loss landscapes of the own training loss around 19 independently trained networks per method
(nonsmooth nonconvex SOCP, L1 penalty). Each panel is a piecewise-planar sheet: the 19 trained
networks, permutation-aligned to one of them (weight matching), are placed on a triangular
lattice, and inside each lattice triangle the surface is the exact plane through its three
networks. Colour: log10 of the loss relative to the best network on the sheet (held-out test
instances), capped at 3 decades. Red dots: trained networks. Pair barrier: lowest-barrier path
between neighbouring networks, relative to their loss; escape barrier: lowest path from an
interior network to any better one. Medians over 3 random assignments of networks to lattice sites.

| model | split | pair barrier (median) | escape barrier (median) |
|---|---|---:|---:|
""" + "\n".join(rows) + "\n"
    open(f"{OUT}/fig1_final_caption.md", "w").write(caption)
    print(caption)


if __name__ == "__main__":
    main()
