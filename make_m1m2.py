"""M1 (SL, rho 10) vs M2 (SL, rho 1e5), both L1 penalty, held-out test split.

Same weights, same directions, only rho changes: the SL training loss
100 * huber(y - y*) + rho * L1 violation is rebuilt from the stored components with
rho = 10 and rho = 1e5 on M1's and on M2's planes (x = top Hessian eigenvector of that
model's held-out loss, y = random filter-normalized direction).
Every surface is min-max normalized (only the shape is compared; Adam ignores scale).

Also prints, per (weights, rho):
  kink   s(2h)/s(h) along x at the weights: ~2 smooth minimum, ~1 V-shaped kink
  pen    share of the loss at the weights coming from the rho * violation term
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

OUT = "figures/landscape/fig1"
RHOS = [("small", 10.0), ("large", 1e5)]


def comps(m, R, kind):
    d = dict(np.load(f"{OUT}/{kind}_test_r{R}_{m}.npz", allow_pickle=True))
    return d["xs"], d["ys"], {k: d[f"{m}/{k}"] for k in d["components"]}


def sl_loss(c, rho):
    return 100 * c["huber"] + rho * c["viol_l1"]


def norm(Z):
    return (Z - Z.min()) / (Z.max() - Z.min())


def kink(xs, Z):
    c = len(xs) // 2
    line = Z[c, :]
    return float(np.mean([((line[c + 2 * g] - line[c]) / 2) / (line[c + g] - line[c]) for g in (-1, 1)]))


def main():
    print("| weights | loss | kink along top direction (2 smooth, 1 V) | kink along random | penalty share at weights |")
    print("|---|---|---:|---:|---:|")
    for m in ["M1", "M2"]:
        xs, ys, c = comps(m, "0.1", "eig")
        for tag, rho in RHOS:
            Z = sl_loss(c, rho)
            jc = len(ys) // 2
            kx = kink(xs, Z)
            ky = kink(ys, Z.T)
            share = rho * c["viol_l1"][jc, jc] / Z[jc, jc]
            print(f"| {m} | SL, rho {rho:g} | {kx:.2f} | {ky:.2f} | {share:.1%} |")

    for R in ["0.1", "0.5"]:
        fig = plt.figure(figsize=(17, 8.5))
        col = 0
        for m in ["M1", "M2"]:
            xs, ys, c = comps(m, R, "eig")
            X, Y = np.meshgrid(xs, ys)
            for tag, rho in RHOS:
                Z = norm(sl_loss(c, rho))
                col += 1
                ax = fig.add_subplot(2, 4, col, projection="3d")
                ax.plot_surface(X, Y, np.log10(Z + 1e-3), cmap=cm.GnBu_r, vmin=-3, vmax=0, shade=False,
                                linewidth=0, antialiased=True, rstride=1, cstride=1)
                ax.set_zlim(-3, 0), ax.set_axis_off(), ax.view_init(elev=30, azim=200)
                own = (m == "M1" and tag == "small") or (m == "M2" and tag == "large")
                ax.set_title(f"{m} weights, SL loss, {tag} " + r"$\rho$" + (" (own)" if own else ""), fontsize=12)
                ax2 = fig.add_subplot(2, 4, 4 + col)
                jc = len(ys) // 2
                ax2.plot(xs, Z[jc, :], label="top Hessian direction")
                ax2.plot(ys, Z[:, jc], label="random direction")
                ax2.set_xlabel("step"), ax2.set_ylim(-0.02, 1.02)
                if col == 1:
                    ax2.set_ylabel("normalized held-out SL loss"), ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT}/m1_vs_m2_r{R}.png", dpi=200)
        plt.close(fig)
        print(f"{OUT}/m1_vs_m2_r{R}.png")


if __name__ == "__main__":
    main()
