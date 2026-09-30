"""Publication version of Fig. 1: four methods on the plane through aligned seeds (0, 1, 2).

Same data and normalization as make_fig1_main.py (cached grids, fig1_plot.surface with norm="method":
log10 of each method's own held-out training loss relative to its best of 10 trained networks),
styled for a two-column paper. Columns: SL soft penalty rho 10, SSL soft penalty rho 10, SL soft
penalty rho 1e5, SL + hard FS rho 1 (collapsed seeds marked). Rows: 3D surface, top view, loss along
the line through s0 and s1. Writes paper_figs/fig1/fig1_pub.pdf and fig1_pub.png.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, patheffects as pe
from matplotlib.lines import Line2D

from fig1_plot import surface

VIEW, SEEDS = "plane", (0, 1, 2)
VARIANTS = {  # name -> (columns, output suffix)
    "rho10": ([("M1", r"$\mathcal{L}_{\mathrm{SL}}$, soft penalty, $\rho = 10$"),
               ("M4", r"$\mathcal{L}_{\mathrm{SSL}}$, soft penalty, $\rho = 10$"),
               ("M2", r"$\mathcal{L}_{\mathrm{SL}}$, soft penalty, $\rho = 10^5$"),
               ("M3r1", r"$\mathcal{L}_{\mathrm{SL}}$, hard FS, $\rho = 1$")], ""),
    "rho1": ([("M1r1", r"$\mathcal{L}_{\mathrm{SL}}$, soft penalty, $\rho = 1$"),
              ("M4r1", r"$\mathcal{L}_{\mathrm{SSL}}$, soft penalty, $\rho = 1$"),
              ("M2", r"$\mathcal{L}_{\mathrm{SL}}$, soft penalty, $\rho = 10^5$"),
              ("M3r1", r"$\mathcal{L}_{\mathrm{SL}}$, hard FS, $\rho = 1$")], "_rho1"),
}
COLS = VARIANTS["rho10"][0]
COLLAPSED = {"M3r1": {1, 3, 8}}  # merit after FS > 10 (m3_candidates.py)
CROP, CAP = (-1.0, 2.0, -0.8, 1.7), 4.0
DROP_3D = 0.07  # move the 3D row down (figure fraction) to close the gap above the top views
CMAP = cm.viridis
OUT = "paper_figs/fig1/fig1_pub"  # + variant suffix

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7.5, "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
GOOD = dict(marker="o", color="#d62728", edgecolors="k", linewidths=0.5)
BAD = dict(marker="X", color="k", edgecolors="w", linewidths=0.5)


def marker(m, n):
    return GOOD  # every trained network is a red dot, collapsed ones included


def merit_min(m):
    """(x, y, z_index) of the minimum of the merit (objective + 1e5 L1 violation, after FS for hard FS) on
    the plane of method m; the index refers to the same grid as the training-loss surface."""
    sm = surface(VIEW, m, "merit", CAP, CROP, None, "triple")
    return np.unravel_index(np.nanargmin(sm["Z"]), sm["Z"].shape)


def panel_3d(ax, s, m):
    X, Y = np.meshgrid(s["xs"], s["ys"])
    surf = ax.plot_surface(X, Y, s["Tc"], cmap=CMAP, vmin=0, vmax=CAP, rstride=2, cstride=2, linewidth=0,
                           antialiased=False, rasterized=True)
    surf.set_rasterized(True)
    ax.computed_zorder = False
    for n, (j, i) in enumerate(s["idx"]):
        ax.scatter([s["xs"][i]], [s["ys"][j]], [s["Tc"][j, i]], s=22, depthshade=False, zorder=10, **marker(m, n))
    jm, im = merit_min(m)
    ax.scatter([s["xs"][im]], [s["ys"][jm]], [s["Tc"][jm, im]], marker="*", s=70, c="w", edgecolors="k",
               linewidths=0.5, depthshade=False, zorder=9)
    ax.set_zlim(0, CAP)
    ax.view_init(elev=38, azim=-62)
    ax.set_box_aspect((1, 1, 0.62), zoom=1.12)
    ax.set_axis_off()  # height is read from the colour scale (shared with the top view)


def panel_2d(ax, s, m):
    X, Y = np.meshgrid(s["xs"], s["ys"])
    cs = ax.contourf(X, Y, s["Tc"], levels=np.linspace(0, CAP, 33), cmap=CMAP, zorder=0)
    ax.set_rasterization_zorder(0.5)  # rasterize the filled contours only (no hairline seams in PDF viewers)
    y0 = s["ys"][int(np.argmin(np.abs(s["ys"])))]
    ax.axhline(y0, color="w", lw=0.8, ls=(0, (3, 2)), path_effects=[pe.withStroke(linewidth=1.8, foreground="k")])
    for n, (j, i) in enumerate(s["idx"]):
        ax.scatter([s["xs"][i]], [s["ys"][j]], s=26, zorder=5, **marker(m, n))
    jm, im = merit_min(m)
    ax.scatter([s["xs"][im]], [s["ys"][jm]], marker="*", s=80, c="w", edgecolors="k", linewidths=0.5, zorder=4)
    ax.set_aspect("equal"), ax.set_axis_off()
    return cs


def barrier(s):
    """Standard linear-interpolation barrier (Frankle et al. 2020) on s0 -> s1, in decades: the largest
    rise of log L above the straight line between its two endpoint values."""
    xs, t = s["xs"], s["T"][int(np.argmin(np.abs(s["ys"])))]
    seg = (xs >= 0) & (xs <= 1)
    a, b = np.interp(0, xs, t), np.interp(1, xs, t)
    return (t[seg] - ((1 - xs[seg]) * a + xs[seg] * b)).max()


def plateau(s):
    """Share of the shown plane that is a high, flat plateau: more than 10x above L* with a slope below
    0.1 decades per unit of ||theta_1 - theta_0|| (little gradient to follow)."""
    gy, gx = np.gradient(s["T"], s["ys"], s["xs"])
    return ((s["T"] > 1) & (np.hypot(gx, gy) < 0.1)).mean()


def slice_curve(s):
    """log10(L / L*) along the row of the plane through s0 and s1 (uncapped)."""
    return s["T"][int(np.argmin(np.abs(s["ys"])))]


def slice_limits(surfs):
    """Shared y-limits and integer ticks for the slice row, from the highest curve of any column
    (rounded up to the next half decade)."""
    top = np.ceil((max(np.nanmax(slice_curve(s)) for s in surfs) + 0.1) * 2) / 2
    return (-0.1, top), list(range(0, int(top) + 1))


def panel_slice(ax, s, m, first, ylim, yticks):
    t = slice_curve(s)
    ax.axvspan(0, 1, color="0.93", lw=0, zorder=0)
    ax.plot(s["xs"], t, color="k", lw=0.9, zorder=2)
    for n, p in enumerate(s["pts"][:2]):
        ax.scatter([p[0]], [np.interp(p[0], s["xs"], t)], s=26, zorder=5, **marker(m, n))
    ax.set_xlim(s["xs"][0], s["xs"][-1]), ax.set_ylim(*ylim)
    ax.set_xticks([0, 1]), ax.set_xticklabels([r"$s_0$", r"$s_1$"])
    ax.set_xlabel(f"barrier {10 ** barrier(s):,.0f}" + r"$\times$", fontsize=7, labelpad=1.5)
    ax.set_yticks(yticks)
    ax.spines[["top", "right"]].set_visible(False)
    if first:
        ax.set_ylabel(r"$\log_{10}(\mathcal{L} / \mathcal{L}^\star)$")
    else:
        ax.set_yticklabels([])


def main(variant="rho10"):
    cols, suffix = VARIANTS[variant]
    out = OUT + suffix
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(3, len(cols) + 1, width_ratios=[1] * len(cols) + [0.045],
                          height_ratios=[1.05, 1, 0.72], hspace=0.04, wspace=0.08,
                          left=0.07, right=0.93, top=0.95, bottom=0.12)
    cs = None
    surfs = [surface(VIEW, m, "own", CAP, CROP, None, "method") for m, _ in cols]
    ylim, yticks = slice_limits(surfs)
    for c, ((m, label), s) in enumerate(zip(cols, surfs)):
        ax3 = fig.add_subplot(gs[0, c], projection="3d")
        panel_3d(ax3, s, m)
        p = ax3.get_position()
        ax3.set_position([p.x0, p.y0 - DROP_3D, p.width, p.height])
        fig.text(p.x0 + p.width / 2, p.y1 - DROP_3D - 0.012, label, ha="center", va="bottom", fontsize=8)
        ax2 = fig.add_subplot(gs[1, c])
        cs = panel_2d(ax2, s, m)
        ax1 = fig.add_subplot(gs[2, c])
        panel_slice(ax1, s, m, c == 0, ylim, yticks)
    cax = fig.add_subplot(gs[0:2, -1])
    cb = fig.colorbar(cs, cax=cax, ticks=[0, 1, 2, 3, 4])
    cb.set_label(r"$\log_{10}(\mathcal{L} / \mathcal{L}^\star)$", labelpad=2)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.08 * pos.height, pos.width, 0.72 * pos.height])
    keys = [Line2D([], [], ls="", marker="o", mfc="#d62728", mec="k", mew=0.5, ms=5, label="trained weights"),
            Line2D([], [], ls="", marker="*", mfc="w", mec="k", mew=0.5, ms=8, label="merit minimum")]
    fig.legend(handles=keys, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.3, columnspacing=2.0)
    fig.savefig(f"{out}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)  # trims the freed top margin
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"{out}.pdf", f"{out}.png")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "rho10")
