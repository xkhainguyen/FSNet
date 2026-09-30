"""Plots and metrics for compute_landscape_compare.py output.

Losses are rebuilt from the stored components:
  L_sl      = 100 * huber(y - y*)     + rho_small * pen2(y)     (model 1)
  L_sl_mid  = 100 * huber(y - y*)     + rho_mid   * pen2(y)     (rho sweep)
  L_sl_high = 100 * huber(y - y*)     + rho_high  * pen2(y)     (model 2)
  L_sl_fs   = 100 * huber(FS(y) - y*) + rho_small * pen2(y)     (model 3)
  L_ssl     = obj(y)                  + rho_small * pen2(y)     (model 4)
  merit     = obj + 1e5 * l1 violation, on y and on FS(y)

Ruggedness metrics are computed on each surface after min-max normalization,
so losses with different scales are comparable:
  n_min      strict local minima on the grid (8-neighborhood)
  nonconvex  fraction of grid points whose finite-difference Hessian has
             lambda_min < -0.1 * |lambda_max| (the ratio map of Li et al.)
  roughness  mean |discrete Laplacian| of the normalized surface
  center     normalized value at the trained weights (0 = grid minimum)
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

SUP_W = 100.0
MERIT_W = 1e5

LOSS_LABELS = {
    "L_sl": r"SL, small $\rho$",
    "L_sl_mid": r"SL, mid $\rho$",
    "L_sl_high": r"SL, high $\rho$",
    "L_sl_fs": r"SL + FS layer, small $\rho$",
    "L_ssl": r"SSL, small $\rho$",
    "merit": "merit (raw)",
    "merit_fs": "merit (after FS)",
}
EXTRA_LABELS = {
    "L_sl_l1_small": r"SL, L1 pen, small $\rho$",
    "L_sl_l1_high": r"SL, L1 pen, high $\rho$",
    "L_ssl_l1_small": r"SSL, L1 pen, small $\rho$",
    "L_ssl_l1_high": r"SSL, L1 pen, high $\rho$",
}


def label(k):
    return LOSS_LABELS.get(k, EXTRA_LABELS.get(k, k))


def build_losses(d, prefix, rho_small, rho_mid, rho_high):
    c = {k: d[f"{prefix}/{k}"] for k in d["components"]}
    L = {
        "L_sl": SUP_W * c["huber"] + rho_small * c["pen2"],
        "L_sl_mid": SUP_W * c["huber"] + rho_mid * c["pen2"],
        "L_sl_high": SUP_W * c["huber"] + rho_high * c["pen2"],
        "L_ssl": c["obj"] + rho_small * c["pen2"],
        "merit": c["obj"] + MERIT_W * c["viol_l1"],
    }
    # L1-penalty variants (pen_type l1), for own-loss plots via --own.
    for tag, rho in [("small", rho_small), ("high", rho_high)]:
        L[f"L_sl_l1_{tag}"] = SUP_W * c["huber"] + rho * c["viol_l1"]
        L[f"L_ssl_l1_{tag}"] = c["obj"] + rho * c["viol_l1"]
    if np.isfinite(c["huber_fs"]).all() and c["huber_fs"].any():
        L["L_sl_fs"] = SUP_W * c["huber_fs"] + rho_small * c["pen2"]
        L["merit_fs"] = c["obj_fs"] + MERIT_W * c["viol_l1_fs"]
    return L


def normalize(Z):
    return (Z - Z.min()) / (Z.max() - Z.min() + 1e-30)


def ruggedness(Z, xs, ys):
    Zn = normalize(Z)
    h = xs[1] - xs[0]
    inner = Zn[1:-1, 1:-1]
    neigh = np.stack([Zn[1 + dj:Zn.shape[0] - 1 + dj, 1 + di:Zn.shape[1] - 1 + di]
                      for dj in (-1, 0, 1) for di in (-1, 0, 1) if (di, dj) != (0, 0)])
    n_min = int((inner[None] < neigh).all(0).sum())
    zxx = (Zn[1:-1, 2:] - 2 * inner + Zn[1:-1, :-2]) / h ** 2
    zyy = (Zn[2:, 1:-1] - 2 * inner + Zn[:-2, 1:-1]) / h ** 2
    zxy = (Zn[2:, 2:] - Zn[2:, :-2] - Zn[:-2, 2:] + Zn[:-2, :-2]) / (4 * h ** 2)
    tr, det = zxx + zyy, zxx * zyy - zxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    lmax, lmin = tr / 2 + disc, tr / 2 - disc
    nonconvex = float((lmin < -0.1 * np.abs(lmax)).mean())
    rough = float(np.abs(zxx + zyy).mean() * h ** 2)
    jc, ic = np.argmin(np.abs(ys)), np.argmin(np.abs(xs))
    return dict(n_min=n_min, nonconvex=nonconvex, roughness=rough, center=float(Zn[jc, ic]))


def argmin_xy(Z, xs, ys):
    j, i = np.unravel_index(np.nanargmin(Z), Z.shape)
    return xs[i], ys[j]


def contour(ax, X, Y, Z, n_levels=25, log=True):
    Zp = Z - Z.min() + 1e-12 * max(1.0, abs(Z.max())) if log else Z
    if log:
        levels = np.geomspace(Zp.min() * 1.0001, Zp.max(), n_levels)
        cs = ax.contourf(X, Y, Zp, levels=levels, norm=colors.LogNorm(), cmap="viridis")
    else:
        cs = ax.contourf(X, Y, Zp, levels=n_levels, cmap="viridis")
    ax.contour(X, Y, Zp, levels=cs.levels, colors="k", linewidths=0.3, alpha=0.4)
    return cs


def plot_3d_row_shared(X, Y, surfaces, titles, path, zlog=True, elev=40, azim=190):
    """3D surfaces on ONE absolute z-axis: relative rise (L - L0) / |L0| at the grid center.

    Unlike plot_3d_row, surfaces are not normalized individually, so a sharper
    loss shows as a steeper bowl. zlog plots log10(1 + rise).
    """
    from matplotlib import cm
    jc, ic = X.shape[0] // 2, X.shape[1] // 2
    R = [(Z - Z[jc, ic]) / abs(Z[jc, ic]) for Z in surfaces]
    if zlog:
        R = [np.log10(1 + np.maximum(r, 0)) for r in R]
    zmax = max(r.max() for r in R)
    fig = plt.figure(figsize=(5 * len(surfaces), 5))
    for n, (r, title) in enumerate(zip(R, titles)):
        ax = fig.add_subplot(1, len(surfaces), n + 1, projection="3d")
        ax.plot_surface(X, Y, r, cmap=cm.GnBu_r, vmin=0, vmax=zmax, shade=False,
                        linewidth=0, antialiased=True, rstride=1, cstride=1)
        ax.set_zlim(0, zmax)
        ax.set_axis_off()
        ax.set_box_aspect(None, zoom=1.2)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title}\nmax rise {r.max() if not zlog else 10 ** r.max() - 1:.3g}", fontsize=13, pad=0)
    fig.subplots_adjust(wspace=0.0, left=0, right=1, top=0.9, bottom=0)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_3d_row(X, Y, surfaces, titles, path, zlog=False, points=None, elev=40, azim=190):
    """3D surfaces in the visualize_weight_to_merit.ipynb style, one shared z range.

    Each surface is min-max normalized; zlog plots log10 of it (floor 1e-4) to
    expose structure near the minimum. points: optional list of (x, y) marked on
    every panel.
    """
    from matplotlib import cm
    fig = plt.figure(figsize=(5 * len(surfaces), 5))
    for n, (Z, title) in enumerate(zip(surfaces, titles)):
        Zp = normalize(Z)
        if zlog:
            Zp = (np.log10(Zp + 1e-4) + 4) / 4
        ax = fig.add_subplot(1, len(surfaces), n + 1, projection="3d")
        ax.plot_surface(X, Y, Zp, cmap=cm.GnBu_r, vmin=0, vmax=1, shade=False,
                        linewidth=0, antialiased=True, rstride=1, cstride=1)
        if points is not None:
            for px, py in points:
                i, j = np.argmin(np.abs(X[0] - px)), np.argmin(np.abs(Y[:, 0] - py))
                ax.scatter([X[0, i]], [Y[j, 0]], [Zp[j, i]], color="r", s=40, depthshade=False)
        ax.set_zlim(0, 1)
        ax.set_axis_off()
        ax.set_box_aspect(None, zoom=1.2)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=13, pad=0)
    fig.subplots_adjust(wspace=0.0, left=0, right=1, top=0.92, bottom=0)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_random(d, models, rho_small, rho_mid, rho_high, outdir, own_loss, ref):
    xs, ys = d["xs"], d["ys"]
    X, Y = np.meshgrid(xs, ys)
    losses = {m: build_losses(d, m, rho_small, rho_mid, rho_high) for m in models}

    # (a) each model's own training loss around its own weights, on identical
    # axes and one shared log color scale of the min-max normalized loss.
    levels = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 25)])
    norm = colors.SymLogNorm(linthresh=1e-4, vmin=0.0, vmax=1.0)
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), sharex=True, sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        Zn = normalize(losses[m][own_loss[m]])
        cs = ax.contourf(X, Y, Zn, levels=levels, norm=norm, cmap="viridis")
        ax.contour(X, Y, Zn, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
        ax.plot(0, 0, "r*", ms=12)
        ax.set_title(f"{m}\n{label(own_loss[m])}", fontsize=11)
        ax.set_xlabel(r"$\alpha$")
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$\beta$")
    fig.colorbar(cs, ax=axes, shrink=0.8, label="normalized own loss")
    fig.savefig(os.path.join(outdir, "own_loss_contours.png"), dpi=200)
    plt.close(fig)

    # (b) all losses on one shared xy: the random plane around the reference model.
    keys = [k for k in LOSS_LABELS if k in losses[ref]]
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for n, k in enumerate(keys):
        Zn = normalize(losses[ref][k])
        ax.contour(X, Y, Zn, levels=np.geomspace(1e-3, 0.3, 6), colors=[cmap(n)], linewidths=1.0)
        ax.plot(*argmin_xy(losses[ref][k], xs, ys), "X", color=cmap(n), ms=11, mec="k",
                label=f"{LOSS_LABELS[k]} min")
    ax.plot(0, 0, "k*", ms=14, label=f"{ref} weights")
    ax.set_xlabel(r"$\alpha$"), ax.set_ylabel(r"$\beta$"), ax.set_aspect("equal")
    ax.set_title(f"All losses on the {ref} plane")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(os.path.join(outdir, f"all_losses_on_{ref}_plane.png"), dpi=200)
    plt.close(fig)

    own_Z = [losses[m][own_loss[m]] for m in models]
    own_titles = [f"{m}\n{label(own_loss[m])}" for m in models]
    for zlog, tag in [(False, "lin"), (True, "log")]:
        plot_3d_row(X, Y, own_Z, own_titles, os.path.join(outdir, f"own_loss_3d_{tag}.png"), zlog=zlog)
        # (b) every loss on the reference model's plane, same xy.
        keys = [k for k in LOSS_LABELS if k in losses[ref]]
        plot_3d_row(X, Y, [losses[ref][k] for k in keys], [LOSS_LABELS[k] for k in keys],
                    os.path.join(outdir, f"all_losses_on_{ref}_plane_3d_{tag}.png"), zlog=zlog, points=[(0, 0)])

    # (a) the same surfaces on one shared absolute scale, so sharpness differences show.
    plot_3d_row_shared(X, Y, own_Z, own_titles, os.path.join(outdir, "own_loss_3d_shared_lin.png"), zlog=False)
    plot_3d_row_shared(X, Y, own_Z, own_titles, os.path.join(outdir, "own_loss_3d_shared_log.png"), zlog=True)

    # (a) 1D slices through the center, normalized.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    jc, ic = np.argmin(np.abs(ys)), np.argmin(np.abs(xs))
    for m in models:
        Zn = normalize(losses[m][own_loss[m]])
        axes[0].plot(xs, Zn[jc, :], label=m)
        axes[1].plot(ys, Zn[:, ic], label=m)
    for ax, lab in zip(axes, [r"$\alpha$ ($\beta=0$)", r"$\beta$ ($\alpha=0$)"]):
        ax.set_xlabel(lab)
        ax.set_ylabel("normalized own loss")
        ax.set_yscale("symlog", linthresh=1e-3)
    axes[0].legend()
    fig.savefig(os.path.join(outdir, "own_loss_slices.png"), dpi=200)
    plt.close(fig)

    # (b) every loss on every model's plane, with each loss's argmin marked.
    names = [k for k in LOSS_LABELS if all(k in losses[m] for m in models)]
    fig, axes = plt.subplots(len(models), len(names), figsize=(3.3 * len(names), 3.1 * len(models)),
                             constrained_layout=True, squeeze=False)
    for r, m in enumerate(models):
        for c, k in enumerate(names):
            ax = axes[r, c]
            contour(ax, X, Y, losses[m][k], n_levels=20)
            ax.plot(0, 0, "r*", ms=10)
            ax.plot(*argmin_xy(losses[m][k], xs, ys), "wx", ms=9, mew=2)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(LOSS_LABELS[k], fontsize=10)
            if c == 0:
                ax.set_ylabel(m, fontsize=10)
    fig.savefig(os.path.join(outdir, "all_losses_all_models.png"), dpi=200)
    plt.close(fig)

    print("\n## Ruggedness of each model's own loss (normalized surface)\n")
    print("| model | loss | n_min | nonconvex | roughness | center |")
    print("|---|---|---:|---:|---:|---:|")
    for m in models:
        r = ruggedness(losses[m][own_loss[m]], xs, ys)
        print(f"| {m} | {own_loss[m]} | {r['n_min']} | {r['nonconvex']:.3f} | {r['roughness']:.2e} | {r['center']:.3f} |")

    print("\n## Argmin of each loss on each model's plane (model sits at (0, 0))\n")
    print("| plane | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for m in models:
        cells = [f"({a:+.2f}, {b:+.2f})" for a, b in (argmin_xy(losses[m][k], xs, ys) for k in names)]
        print(f"| {m} | " + " | ".join(cells) + " |")

    print("\n## Cross-evaluation at the trained weights (rows: model, cols: loss)\n")
    print("| model | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    for m in models:
        print(f"| {m} | " + " | ".join(f"{losses[m][k][jc, ic]:.4g}" for k in names) + " |")


def plot_plane(d, rho_small, rho_mid, rho_high, outdir):
    xs, ys, pts = d["xs"], d["ys"], d["points"]
    names_pts = list(d["names"])
    X, Y = np.meshgrid(xs, ys)
    L = build_losses(d, "plane", rho_small, rho_mid, rho_high)
    keys = [k for k in LOSS_LABELS if k in L]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.8), constrained_layout=True)
    for ax, k in zip(axes, keys):
        contour(ax, X, Y, L[k])
        for (px, py), n in zip(pts, names_pts):
            ax.plot(px, py, "o", mfc="w", mec="k", ms=7)
            ax.annotate(n, (px, py), textcoords="offset points", xytext=(4, 4), fontsize=8, color="w")
        ax.plot(*argmin_xy(L[k], xs, ys), "rx", ms=10, mew=2)
        ax.set_title(LOSS_LABELS[k], fontsize=11)
        ax.set_aspect("equal")
    fig.savefig(os.path.join(outdir, "plane_all_losses.png"), dpi=200)
    plt.close(fig)
    for zlog, tag in [(False, "lin"), (True, "log")]:
        plot_3d_row(X, Y, [L[k] for k in keys], [LOSS_LABELS[k] for k in keys],
                    os.path.join(outdir, f"plane_all_losses_3d_{tag}.png"), zlog=zlog, points=pts)

    print("\n## Plane through " + ", ".join(names_pts) + "\n")
    print("| loss | argmin | " + " | ".join(f"at {n}" for n in names_pts) + " |")
    print("|---|---|" + "---:|" * len(names_pts))
    for k in keys:
        a, b = argmin_xy(L[k], xs, ys)
        vals = []
        for px, py in pts:
            i, j = np.argmin(np.abs(xs - px)), np.argmin(np.abs(ys - py))
            vals.append(f"{L[k][j, i]:.4g}")
        print(f"| {k} | ({a:+.2f}, {b:+.2f}) | " + " | ".join(vals) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="one or more random-mode files (merged), or one plane-mode file")
    ap.add_argument("--rho_small", type=float, default=10.0)
    ap.add_argument("--rho_mid", type=float, default=1e3)
    ap.add_argument("--rho_high", type=float, default=1e5)
    ap.add_argument("--ref", default="sl_small", help="model whose plane hosts the all-loss overlay")
    ap.add_argument("--own", action="append", default=[],
                    help="model=loss_key, e.g. sl_small=L_sl (random mode)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    files = [dict(np.load(f, allow_pickle=True)) for f in args.npz]
    d = files[0]
    for other in files[1:]:
        assert np.allclose(other["xs"], d["xs"]), "grids differ"
        d.update({k: v for k, v in other.items() if "/" in k})
        d["names"] = np.concatenate([d["names"], other["names"]])
    outdir = args.outdir or os.path.splitext(args.npz[0])[0]
    os.makedirs(outdir, exist_ok=True)
    if str(d["mode"]) == "random":
        models = list(d["names"])
        own = dict(o.split("=", 1) for o in args.own)
        own = {m: own.get(m, "L_sl") for m in models}
        ref = args.ref if args.ref in models else models[0]
        plot_random(d, models, args.rho_small, args.rho_mid, args.rho_high, outdir, own, ref)
    else:
        plot_plane(d, args.rho_small, args.rho_mid, args.rho_high, outdir)
    print(f"\nFigures in {outdir}")


if __name__ == "__main__":
    main()
