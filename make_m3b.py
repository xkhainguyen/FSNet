"""M3b (SL + FS truncated to 10 iterations, rho 10) vs M1 (SL, rho 10): does FS now shape the
training loss? Held-out, radius-1 random plane (seed 0) and the wide plane through seeds 0-2."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

OUT = "figures/landscape/fig1"


def load(f, prefix):
    d = dict(np.load(f, allow_pickle=True))
    return d, {k: d[f"{prefix}/{k}"] for k in d["components"]}


r1, c1 = load(f"{OUT}/random_test_r1.0_M1.npz", "M1")
rb, cb = load(f"{OUT}/random_test_r1.0_M3b.npz", "M3b")
L1 = 100 * c1["huber"] + 10 * c1["viol_l1"]
Lb = 100 * cb["huber_fs"] + 10 * cb["viol_l1"]          # what M3b trains on
Lb_raw = 100 * cb["huber"] + 10 * cb["viol_l1"]          # same weights, no FS
rel = np.abs(Lb - Lb_raw) / Lb_raw
jc = len(rb["xs"]) // 2
print(f"M3b at weights: 100*huber(FS10(y)-y*) = {100*cb['huber_fs'][jc,jc]:.2f}, 100*huber(y-y*) = {100*cb['huber'][jc,jc]:.2f}, "
      f"||FS10(y)-y||^2 = {cb['dist_fs'][jc,jc]:.3f}, violation raw / after FS10 = {cb['viol_l1'][jc,jc]:.3g} / {cb['viol_l1_fs'][jc,jc]:.3g}")
print(f"effect of FS on M3b's loss over the +-1 plane: median {np.median(rel):.2%}, 99th pct {np.percentile(rel, 99):.2%}, max {rel.max():.1%}")

fig = plt.figure(figsize=(16, 6.5))
for n, (Z, title) in enumerate([(L1, r"M1: SL, small $\rho$"), (Lb, r"M3b: SL + FS (10 iters), small $\rho$")]):
    X, Y = np.meshgrid(r1["xs"], r1["ys"])
    T = np.log10(Z)
    ax = fig.add_subplot(1, 2, n + 1, projection="3d")
    ax.plot_surface(X, Y, T, cmap=cm.viridis, rstride=1, cstride=1, linewidth=0.05, edgecolor="k", alpha=0.95)
    ax.scatter([0], [0], [T[jc, jc]], color="r", s=60, depthshade=False)
    ax.view_init(elev=28, azim=-60), ax.set_title(title, fontsize=14), ax.set_zlabel("log10 held-out training loss")
fig.tight_layout()
fig.savefig(f"{OUT}/m1_vs_m3b_random3d_r1.0.png", dpi=180)
print(f"{OUT}/m1_vs_m3b_random3d_r1.0.png")
