"""Fig. 1 plotting library: loads the cached grids in paper_figs/fig1/data/ (fig1_bundle.py) and
draws landscape panels. No torch, no GPU. Used by fig1_layout.ipynb and make_fig1_main.py.

Views: straight (10-seed flat sheet), curved (10-seed Bezier sheet), plane / plane_t345 /
plane_t678 (wide plane through aligned seeds 0-1-2, 3-4-5, 6-7-8), hex19 (19-seed flat sheet,
no M3f). Loss kinds: "own" (each method's training loss, fig1_common.own_loss) or "merit"
(obj + 1e5 * L1 violation; after FS for M3f). Panel kinds: "3d", "2d", "slice", "basins".
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from numpy.lib.stride_tricks import sliding_window_view

from fig1_common import own_loss

DATA = "paper_figs/fig1/data"
VIEWS = {v: dict(np.load(f"{DATA}/{v}.npz")) for v in ["straight", "curved", "plane", "plane_t345", "plane_t678", "hex19", "collapse"]
         if os.path.exists(f"{DATA}/{v}.npz")}


# Exact loss / merit at every trained network (seed_losses.py), for norm="method".
SEED_LOSSES = json.load(open(f"{DATA}/seed_losses.json")) if os.path.exists(f"{DATA}/seed_losses.json") else {}


def method_ref(m, kind):
    """Best (lowest) value of this method's own loss or merit over its 10 trained networks."""
    rows = SEED_LOSSES[m.split("@")[0]]
    return min(r["own" if kind == "own" else "merit"] for r in rows)


def available():
    """Which methods each cached view holds."""
    return {v: sorted({k.split("/")[0] for k in d}) for v, d in VIEWS.items()}

LABEL = {"M1": r"SL, soft penalty, $\rho = 10^1$", "M2": r"SL, soft penalty, $\rho = 10^5$",
         "M3f": "SL, hard FS", "M4": r"SSL, soft penalty, $\rho = 10^1$",
         "M3r1": r"SL, hard FS, $\rho = 1$", "M3r08": r"SL, hard FS, $\rho = 0.8$",
         "M1r1": r"SL, soft penalty, $\rho = 1$", "M4r1": r"SSL, soft penalty, $\rho = 1$"}


def loss(view, m, kind="own"):
    d = VIEWS[view]
    c = {k.split("/", 1)[1]: d[k].astype(np.float64) for k in d if k.startswith(m + "/")}
    base = m.split("@")[0]  # "M3r1@A": method M3r1 on seed triple A (collapse view)
    if kind == "own":
        return own_loss(base, c)
    if kind == "merit":  # after FS for every hard-FS method, raw otherwise
        return c["obj_fs"] + 1e5 * c["viol_l1_fs"] if base.startswith("M3") else c["obj"] + 1e5 * c["viol_l1"]
    raise ValueError(kind)


def local_minima(Z, k=2):
    """Grid points lower than every other point in their (2k+1)^2 window; off-sheet = +inf."""
    Zs = np.where(np.isfinite(Z), Z, np.inf)
    w = sliding_window_view(np.pad(Zs, k, constant_values=np.inf), (2 * k + 1, 2 * k + 1))
    others = np.delete(w.reshape(*Zs.shape, -1), (2 * k + 1) ** 2 // 2, axis=-1)
    is_min = np.isfinite(Zs) & (Zs[..., None] < others).all(-1)
    is_min[:k, :] = is_min[-k:, :] = is_min[:, :k] = is_min[:, -k:] = False
    return np.nonzero(is_min)


def surface(view, m, kind="own", cap=3.0, crop=None, minima_below=None, norm="triple"):
    """log10(L / L_best_seed), clipped to [0, cap]; seed grid indices; local minima.
    crop: (x0, x1, y0, y1) in plane coordinates, applied identically to every method.
    minima_below: keep only local minima within this many decades of the best seed.
    norm: "triple" = relative to the best of the three networks on this plane; "method" = relative
    to the method's best of its 10 trained networks (exact values, seed_losses.py).
    If that reference value is <= 0 the panel shows log10(L - L* + 1) and s["shifted"] is True."""
    d = VIEWS[view]
    xs, ys, pts = d[f"{m}/xs"], d[f"{m}/ys"], d[f"{m}/points"]
    Z = loss(view, m, kind)
    if crop is not None:
        ci = (xs >= crop[0]) & (xs <= crop[1])
        cj = (ys >= crop[2]) & (ys <= crop[3])
        xs, ys, Z = xs[ci], ys[cj], Z[np.ix_(cj, ci)]
    inside = np.isfinite(Z)
    Jg, Ig = np.nonzero(inside)  # nearest on-sheet grid point to each seed
    idx = [(int(Jg[k]), int(Ig[k])) for k in (np.argmin((xs[Ig] - p[0]) ** 2 + (ys[Jg] - p[1]) ** 2) for p in pts)]
    ref = method_ref(m, kind) if norm == "method" else min(Z[k] for k in idx)
    shifted = ref <= 0  # a ratio needs L* > 0; losses that go negative (SSL at small rho) are plotted as
    if shifted:         # log10(L - L* + 1) instead, with L* the same reference value
        Z, ref = Z - ref + 1.0, 1.0
    T = np.log10(np.maximum(Z, 1e-12) / ref)
    jm, im = local_minima(Z)
    if minima_below is not None:
        keep = T[jm, im] < minima_below
        jm, im = jm[keep], im[keep]
    return dict(method=m, xs=xs, ys=ys, pts=pts, Z=Z, T=T, Tc=np.where(inside, np.clip(T, 0, cap), np.nan), idx=idx, minima=(jm, im), shifted=shifted)


STYLE = dict(cap=3.0, elev=42, azim=-65, stride=2, mesh=True, minima=True, seed_labels=False,
             label_size=16, cmap="viridis", crop=None, minima_below=None)


def seed_style(s, n, st):
    """Marker kwargs for the n-th trained network of panel s: red dot, or a black X if collapsed."""
    if n in st.get("collapsed", {}).get(s.get("method"), ()):
        return dict(marker="X", color="k", edgecolors="w", s=110)
    return dict(marker="o", color="r", edgecolors="k", s=50)


def draw3d(ax, s, st):
    X, Y = np.meshgrid(s["xs"], s["ys"])
    kw = dict(linewidth=0.02, edgecolor="k") if st["mesh"] else dict(linewidth=0, antialiased=False)
    ax.plot_surface(X, Y, s["Tc"], cmap=st["cmap"], vmin=0, vmax=st["cap"], rstride=st["stride"],
                    cstride=st["stride"], alpha=0.92, **kw)
    if st["minima"]:
        jm, im = s["minima"]
        ax.scatter(s["xs"][im], s["ys"][jm], s["Tc"][jm, im], color="w", edgecolors="k", s=26, depthshade=False)
    ax.computed_zorder = False  # draw the trained networks on top even where a wall is in front
    for n, (j, i) in enumerate(s["idx"]):
        ax.scatter([s["xs"][i]], [s["ys"][j]], [s["Tc"][j, i]], depthshade=False, zorder=10, **seed_style(s, n, st))
    ax.set_zlim(0, st["cap"]), ax.set_zticks(range(int(st["cap"]) + 1)), ax.view_init(elev=st["elev"], azim=st["azim"])
    if not st.get("xyticks", True):
        ax.set_xticklabels([]), ax.set_yticklabels([])


def draw2d(ax, s, st):
    X, Y = np.meshgrid(s["xs"], s["ys"])
    cs = ax.contourf(X, Y, np.ma.masked_invalid(s["Tc"]), levels=np.linspace(0, st["cap"], 31), cmap=st["cmap"])
    if st["minima"]:
        jm, im = s["minima"]
        ax.plot(s["xs"][im], s["ys"][jm], "o", mfc="w", mec="k", ms=5)
    for n, (j, i) in enumerate(s["idx"]):
        k = seed_style(s, n, st)
        ax.scatter([s["xs"][i]], [s["ys"][j]], marker=k["marker"], c=k["color"], edgecolors=k["edgecolors"],
                   s=k["s"] * 1.3, zorder=5)
        if st["seed_labels"]:
            ax.annotate(f"s{n}", (s["xs"][i], s["ys"][j]), textcoords="offset points", xytext=(4, 4), color="w")
    ax.set_aspect("equal"), ax.set_axis_off()
    return cs


def draw_slice(ax, s, st, ymax=6.0):
    """Loss along the plane's x axis (the line through s0 = (0, 0) and s1 = (1, 0)), uncapped."""
    j0 = int(np.argmin(np.abs(s["ys"])))
    t = s["T"][j0]
    ax.axvspan(0, 1, color="0.92", zorder=0)  # the segment between the two networks
    ax.plot(s["xs"], np.minimum(t, ymax), color="k", lw=1.3)
    over = t > ymax
    if over.any():  # values above the axis: mark them instead of hiding them
        ax.plot(s["xs"][over], np.full(over.sum(), ymax), "^", color="k", ms=4)
    for n, p in enumerate(s["pts"][:2]):
        k = seed_style(s, n, st)
        ax.scatter([p[0]], [np.interp(p[0], s["xs"], t)], marker=k["marker"], c=k["color"], edgecolors=k["edgecolors"],
                   s=k["s"] * 1.3, zorder=5)
    ax.set_ylim(-0.1, ymax + 0.2), ax.set_xlim(s["xs"][0], s["xs"][-1])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([0, 1]), ax.set_xticklabels(["s0", "s1"])
    seg = (s["xs"] >= 0) & (s["xs"] <= 1)
    climb = t[seg].max() - max(np.interp(0, s["xs"], t), np.interp(1, s["xs"], t))
    ax.text(0.5, 0.97, f"barrier s0 to s1: {10 ** climb:,.0f}" + r"$\times$", transform=ax.transAxes,
            ha="center", va="top", fontsize=st["label_size"] - 3)


def basins(Z):
    """Label every grid point by the local minimum that grid steepest descent (8-connected) reaches."""
    from make_rugged import basins as _basins
    return _basins(np.where(np.isfinite(Z), Z, np.inf))


def draw_basins(ax, s, st):
    """Basins of attraction in this slice; thin lines: contours of log10(L / L_best)."""
    X, Y = np.meshgrid(s["xs"], s["ys"])
    lab = basins(s["Z"])
    order = np.random.default_rng(1).permutation(lab.max() + 1)
    ax.pcolormesh(X, Y, np.ma.masked_invalid(np.where(np.isfinite(s["Z"]), order[lab] % 20, np.nan)),
                  cmap=plt.get_cmap("tab20", 20), shading="auto")
    ax.contour(X, Y, np.ma.masked_invalid(s["Tc"]), levels=np.linspace(0, st["cap"], 13), colors="k",
               linewidths=0.3, alpha=0.5)
    for j, i in s["idx"]:
        ax.plot(s["xs"][i], s["ys"][j], "o", mfc="r", mec="k", ms=8)
    ax.set_aspect("equal"), ax.set_axis_off()
    sizes = np.bincount(lab[np.isfinite(s["Z"])].ravel())
    sizes = sizes[sizes > 0] / sizes.sum()
    return len(sizes), sizes.max()


def figure(rows, methods, style=None, path=None, col_w=6.5, row_h=5.5, label_row=None):
    """rows: list of (view, "3d" | "2d" | "slice" | "basins", loss kind). Method labels go on label_row (default: first 2D row)."""
    st = {**STYLE, **(style or {})}
    fig = plt.figure(figsize=(col_w * len(methods), row_h * len(rows)))
    gs = fig.add_gridspec(len(rows), len(methods), hspace=0.08, wspace=0.05)
    if label_row is None:
        label_row = next((r for r, row in enumerate(rows) if row[1] == "2d"), 0)
    cs, axes2d = None, []
    for r, (view, kind3, lk) in enumerate(rows):
        for c, m in enumerate(methods):
            if f"{m}/xs" not in VIEWS.get(view, {}):
                continue
            s = surface(view, m, lk, st["cap"], st["crop"], st["minima_below"], st.get("norm", "triple"))
            if kind3 == "3d":
                ax = fig.add_subplot(gs[r, c], projection="3d")
                draw3d(ax, s, st)
            elif kind3 == "slice":
                ax = fig.add_subplot(gs[r, c])
                draw_slice(ax, s, st, st.get("slice_ymax", 6.0))
                if c == 0:
                    ax.set_ylabel(r"$\log_{10}(L / L_{\mathrm{best}})$ along s0-s1", fontsize=st["label_size"] - 4)
                else:
                    ax.set_yticklabels([])
            elif kind3 == "basins":
                ax = fig.add_subplot(gs[r, c])
                nb, big = draw_basins(ax, s, st)
                ax.text(0.5, -0.04, f"{nb} basins, largest {big:.0%}", transform=ax.transAxes, ha="center",
                        va="top", fontsize=st["label_size"] - 3)
            else:
                ax = fig.add_subplot(gs[r, c])
                cs = draw2d(ax, s, st)
                if st.get("mark_slice"):
                    from matplotlib import patheffects as pe
                    ax.axhline(s["ys"][int(np.argmin(np.abs(s["ys"])))], color="w", lw=1.4, ls="--",
                               path_effects=[pe.withStroke(linewidth=3, foreground="k")])
                axes2d.append(ax)
            if r == label_row:
                ax.legend(handles=[Line2D([], [], ls="", label=st.get("labels", {}).get(m) or LABEL[m.split("@")[0]])], loc="upper center",
                          bbox_to_anchor=(0.5, 1.13), handlelength=0, handletextpad=0,
                          fontsize=st["label_size"], frameon=True)
    keys = [Line2D([], [], ls="", marker="o", mfc="r", mec="k", ms=10, label="trained models")]
    if st.get("collapsed"):
        keys[0] = Line2D([], [], ls="", marker="o", mfc="r", mec="k", ms=10, label="trained, good")
        keys.append(Line2D([], [], ls="", marker="X", mfc="k", mec="w", ms=12, label="trained, collapsed (merit > 10)"))
    if st.get("mark_slice") and any(r[1] == "slice" for r in rows):
        keys.append(Line2D([], [], color="0.4", ls="--", lw=1.5, label="line through s0 and s1 (bottom row)"))
    if st["minima"]:
        keys.append(Line2D([], [], ls="", marker="o", mfc="w", mec="k", ms=7, label="local minima"))
    fig.legend(handles=keys, loc="lower center", ncol=len(keys), bbox_to_anchor=(0.45, 0.0), fontsize=14, frameon=False)
    if cs is not None:
        fig.colorbar(cs, ax=axes2d, shrink=0.8, pad=0.01, ticks=range(int(st["cap"]) + 1),
                     label=st.get("cbar_label", r"$\log_{10}(L / L_{\mathrm{best}})$, held-out"))
    if path:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(path)
    return fig

