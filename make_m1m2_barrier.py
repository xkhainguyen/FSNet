"""M1 vs M2 barriers on the plane through 3 seeds (held-out SL loss, L1 penalty, 121x121).

Top: contours of log10(L / L_min) on ONE shared color scale (decades above the model's
     best seed), with the lowest-barrier path between each pair of seeds.
Bottom: loss along those paths, log10(L / max(end losses)) vs normalized path length,
        M1 and M2 on the same axes. The peak is the barrier an optimizer must climb.
"""
import heapq

import numpy as np
import matplotlib.pyplot as plt

from make_seedplanes import sl

OUT = "figures/landscape/fig1"
MODELS = [("M1", 10.0, r"M1: SL, small $\rho$"), ("M2", 1e5, r"M2: SL, large $\rho$")]
PAIRS = [(0, 1), (0, 2), (1, 2)]


def minimax_path(Z, a, b):
    best = np.full(Z.shape, np.inf)
    prev = {}
    best[a] = Z[a]
    pq = [(Z[a], a)]
    while pq:
        v, (j, i) = heapq.heappop(pq)
        if (j, i) == b:
            break
        if v > best[j, i]:
            continue
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                jj, ii = j + dj, i + di
                if (dj or di) and 0 <= jj < Z.shape[0] and 0 <= ii < Z.shape[1]:
                    nv = max(v, Z[jj, ii])
                    if nv < best[jj, ii]:
                        best[jj, ii] = nv
                        prev[(jj, ii)] = (j, i)
                        heapq.heappush(pq, (nv, (jj, ii)))
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), gridspec_kw={"height_ratios": [1.15, 1]})
    levels = np.linspace(0, 3, 31)
    colors_pair = ["tab:red", "tab:orange", "tab:pink"]
    print("| model | pair | barrier (x end loss) | decades |")
    print("|---|---|---:|---:|")
    for n, (m, rho, title) in enumerate(MODELS):
        d = dict(np.load(f"{OUT}/seedplane_fine_test_{m}.npz", allow_pickle=True))
        xs, ys, pts = d["xs"], d["ys"], d["points"]
        Z = sl(d, "plane", rho)
        idx = [(int(np.argmin(np.abs(ys - p[1]))), int(np.argmin(np.abs(xs - p[0])))) for p in pts]
        zmin = min(Z[k] for k in idx)
        X, Y = np.meshgrid(xs, ys)
        ax = axes[0, n]
        cs = ax.contourf(X, Y, np.log10(Z / zmin), levels=levels, cmap="viridis", extend="max")
        ax.contour(X, Y, np.log10(Z / zmin), levels=levels[::3], colors="k", linewidths=0.3, alpha=0.5)
        for (a, b), col in zip(PAIRS, colors_pair):
            path = minimax_path(Z, idx[a], idx[b])
            px, py = [xs[i] for _, i in path], [ys[j] for j, _ in path]
            ax.plot(px, py, "-", color=col, lw=1.6)
            prof = np.array([Z[p] for p in path]) / max(Z[idx[a]], Z[idx[b]])
            s = np.linspace(0, 1, len(prof))
            axes[1, n].plot(s, np.log10(prof), color=col, lw=2, label=f"s{a} to s{b}")
            print(f"| {m} | s{a}-s{b} | {prof.max():.1f} | {np.log10(prof.max()):.2f} |")
        for p, lab in zip(pts, ["s0", "s1", "s2"]):
            ax.plot(*p, "o", mfc="w", mec="k", ms=9)
            ax.annotate(lab, p, textcoords="offset points", xytext=(6, 6), color="w", fontsize=12)
        ax.set_title(title + ": plane through 3 seeds", fontsize=14)
        ax.set_aspect("equal")
    fig.colorbar(cs, ax=axes[0, :], label=r"$\log_{10}(L / L_{\min})$, shared scale (decades)", shrink=0.85)
    for n, (m, _, title) in enumerate(MODELS):
        ax = axes[1, n]
        ax.set_ylim(-0.2, 2.6), ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("position along lowest-barrier path"), ax.set_title(title + ": barrier between seeds", fontsize=13)
        ax.legend(fontsize=10)
    axes[1, 0].set_ylabel(r"$\log_{10}$(loss / end loss)")
    fig.savefig(f"{OUT}/m1_vs_m2_barriers.png", dpi=200, bbox_inches="tight")
    print(f"{OUT}/m1_vs_m2_barriers.png")


if __name__ == "__main__":
    main()
