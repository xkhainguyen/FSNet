"""Objective-mismatch figure: what each method minimizes vs the merit it is judged on.

Same four columns and plane as fig1_pub_rho1 (seeds 0, 1, 2; SL soft penalty rho 1, SSL soft penalty
rho 1, SL soft penalty rho 1e5, SL hard FS rho 1). Rows:
  1. training loss, log10(L / L*) (the method's own loss; SSL rho 1 shifted, as in fig1_pub_rho1)
  2. merit = objective + 1e5 * L1 constraint violation (after the FS layer for hard FS), log10, on ONE
     absolute scale for all columns, star = merit minimum on the plane
  3. merit at the 10 trained networks of each method (symlog axis), against two references computed on
     100 held-out instances (label_multistart.py): the SL labels (one IPOPT solve from zero; solid line, -2.4)
     and the best of 20 IPOPT starts per instance (dashed line, -4.3)
No legend: the star, the two lines and the dots are explained in fig1_mismatch_caption.md.
Writes paper_figs/fig1/fig1_mismatch.pdf and .png.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from fig1_plot import surface
from make_fig1_pub import VARIANTS, CROP, VIEW, GOOD  # importing it also sets the rcParams

COLS = VARIANTS["rho1"][0]
MERIT_CAP = 8.0  # log10; the FS-divergent region reaches 1e18
OUT = "paper_figs/fig1/fig1_mismatch"
SEED_LOSSES = json.load(open("paper_figs/fig1/data/seed_losses.json"))


def marker(m, n):  # every trained network is a red dot (collapsed ones included; see the strip row)
    return GOOD


def references():
    r = json.load(open("figures/landscape/fig1/label_multistart.json"))
    labels = np.mean([x["label_obj"] for x in r])  # labels are feasible to 1e-8: merit = objective
    best = np.mean([min(s["obj"] for s in x["starts"] if s["viol"] < 1e-6) for x in r])
    return labels, best


def panel_loss(ax, s, m):
    X, Y = np.meshgrid(s["xs"], s["ys"])
    cs = ax.contourf(X, Y, s["Tc"], levels=np.linspace(0, 4, 33), cmap=cm.viridis, zorder=0)
    ax.set_rasterization_zorder(0.5)
    for n, (j, i) in enumerate(s["idx"]):
        ax.scatter([s["xs"][i]], [s["ys"][j]], s=26, zorder=5, **marker(m, n))
    ax.set_aspect("equal"), ax.set_axis_off()
    return cs


def panel_merit(ax, s, m):
    X, Y = np.meshgrid(s["xs"], s["ys"])
    lm = np.clip(np.log10(np.maximum(s["Z"], 1e-3)), 0, MERIT_CAP)
    cs = ax.contourf(X, Y, lm, levels=np.linspace(0, MERIT_CAP, 33), cmap=cm.magma, zorder=0)
    ax.set_rasterization_zorder(0.5)
    for n, (j, i) in enumerate(s["idx"]):
        ax.scatter([s["xs"][i]], [s["ys"][j]], s=26, zorder=5, **marker(m, n))
    jm, im = np.unravel_index(np.nanargmin(s["Z"]), s["Z"].shape)
    ax.scatter([s["xs"][im]], [s["ys"][jm]], marker="*", s=90, c="w", edgecolors="k", linewidths=0.6, zorder=6)
    ax.set_aspect("equal"), ax.set_axis_off()
    return cs


def panel_strip(ax, m, refs, first):
    merits = np.array([r["merit"] for r in SEED_LOSSES[m]])
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.25, 0.25, 10)
    ax.scatter(x, merits, s=16, zorder=5, **GOOD)
    for v, ls in zip(refs, ["-", (0, (3, 2))]):
        ax.axhline(v, color="0.35", lw=0.8, ls=ls, zorder=1)
    ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    ax.set_yscale("symlog", linthresh=10, linscale=0.6)
    ax.set_ylim(-8, 3e7), ax.set_xlim(-0.6, 0.6), ax.set_xticks([])
    ax.set_yticks([-10, 0, 10, 1e3, 1e5, 1e7])
    ax.set_yticklabels([r"$-10$", "0", "10", r"$10^3$", r"$10^5$", r"$10^7$"])
    ax.set_xlabel(f"median {np.median(merits):.1e}".replace("e+0", "e").replace("e+", "e"), fontsize=7, labelpad=2)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    if first:
        ax.set_ylabel("merit at the 10\ntrained networks")
    else:
        ax.set_yticklabels([])


def main():
    refs = references()
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(3, len(COLS) + 1, width_ratios=[1] * len(COLS) + [0.045],
                          height_ratios=[0.8, 0.8, 0.78], hspace=0.05, wspace=0.08,
                          left=0.085, right=0.93, top=0.93, bottom=0.085)
    cs1 = cs2 = None
    for c, (m, label) in enumerate(COLS):
        s_loss = surface(VIEW, m, "own", 4.0, CROP, None, "method")
        s_mer = surface(VIEW, m, "merit", 99, CROP, None, "triple")
        ax1 = fig.add_subplot(gs[0, c])
        cs1 = panel_loss(ax1, s_loss, m)
        p = ax1.get_position()
        fig.text(p.x0 + p.width / 2, p.y1 + 0.012, label, ha="center", va="bottom", fontsize=8)
        ax2 = fig.add_subplot(gs[1, c])
        cs2 = panel_merit(ax2, s_mer, m)
        ax3 = fig.add_subplot(gs[2, c])
        panel_strip(ax3, m, refs, c == 0)
    for row, cs, label in [(0, cs1, r"$\log_{10}(\mathcal{L}/\mathcal{L}^\star)$"), (1, cs2, r"$\log_{10}$ merit")]:
        cax = fig.add_subplot(gs[row, -1])
        cb = fig.colorbar(cs, cax=cax, ticks=[0, 2, 4] if row == 0 else [0, 2, 4, 6, 8])
        cb.set_label(label, labelpad=2)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(width=0.5, length=2)
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 + 0.1 * pos.height, pos.width, 0.75 * pos.height])
    fig.text(0.012, 0.74, "training loss\n(what is minimized)", rotation=90, ha="center", va="center", fontsize=8)
    fig.text(0.012, 0.50, "merit: objective +\n$10^5\\times$ violation", rotation=90, ha="center", va="center", fontsize=8)
    fig.savefig(f"{OUT}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{OUT}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"{OUT}.pdf", f"{OUT}.png")


if __name__ == "__main__":
    main()
