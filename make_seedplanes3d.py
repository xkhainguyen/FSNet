"""3D planes through three independently trained seeds, M1-M4 (L1 penalty, held-out).

z = log10(L / L_best_seed), capped at 3 decades, same z axis for all panels, so pits
(local minima) and the barriers between them are comparable across models.
M4 (SSL) loss can be negative: it is shifted so its grid minimum is 1 (noted in the title).
Printed: lowest-barrier path between each pair of seeds (make_m1m2_barrier.minimax_path).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from make_m1m2_barrier import minimax_path

OUT = "figures/landscape/fig1"
MODELS = [("M1", r"M1: SL, small $\rho$"), ("M2", r"M2: SL, large $\rho$"),
          ("M3", r"M3: SL + FS, small $\rho$"), ("M4", r"M4: SSL, small $\rho$")]
CAP = 3.0


def own(m, d):
    c = {k: d[f"plane/{k}"] for k in d["components"]}
    return {"M1": 100 * c["huber"] + 10 * c["viol_l1"], "M2": 100 * c["huber"] + 1e5 * c["viol_l1"],
            "M3": 100 * c["huber_fs"] + 10 * c["viol_l1"], "M4": c["obj"] + 10 * c["viol_l1"]}[m]


def main():
    fig = plt.figure(figsize=(22, 6.2))
    print("| model | grid | barrier s0-s1 | barrier s0-s2 | barrier s1-s2 | (x end loss, lowest path in the plane) |")
    print("|---|---|---:|---:|---:|---|")
    n = 0
    for m, title in MODELS:
        f = f"{OUT}/seedplane_fine_test_{m}.npz"
        if not os.path.exists(f):
            continue
        n += 1
        d = dict(np.load(f, allow_pickle=True))
        xs, ys, pts = d["xs"], d["ys"], d["points"]
        Z = own(m, d)
        shifted = Z.min() <= 0
        if shifted:
            Z = Z - Z.min() + 1.0
        idx = [(int(np.argmin(np.abs(ys - p[1]))), int(np.argmin(np.abs(xs - p[0])))) for p in pts]
        zbest = min(Z[k] for k in idx)
        bars = [max(Z[p] for p in minimax_path(Z, idx[a], idx[b])) / max(Z[idx[a]], Z[idx[b]])
                for a, b in [(0, 1), (0, 2), (1, 2)]]
        print(f"| {m} | {len(xs)}x{len(ys)} | " + " | ".join(f"{v:.1f}" for v in bars) + " | |")
        T = np.minimum(np.log10(Z / zbest), CAP)
        X, Y = np.meshgrid(xs, ys)
        ax = fig.add_subplot(1, 4, n, projection="3d")
        ax.plot_surface(X, Y, T, cmap=cm.viridis, vmin=0, vmax=CAP, rstride=1, cstride=1,
                        linewidth=0.03, edgecolor="k", alpha=0.95)
        for (j, i), lab in zip(idx, ["s0", "s1", "s2"]):
            ax.scatter([xs[i]], [ys[j]], [T[j, i]], color="r", s=60, depthshade=False)
            ax.text(xs[i], ys[j], T[j, i] + 0.1, lab, color="r", fontsize=11)
        ax.set_zlim(-0.1, CAP), ax.view_init(elev=38, azim=-70)
        ax.set_title(title + (" (shifted)" if shifted else "") + f"\nmin barrier between seeds {min(bars):.0f}x",
                     fontsize=12)
        if n == 1:
            ax.set_zlabel(r"$\log_{10}(L / L_{\mathrm{best\ seed}})$")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_seedplanes3d.png", dpi=180)
    print(f"{OUT}/fig1_seedplanes3d.png")


if __name__ == "__main__":
    main()
