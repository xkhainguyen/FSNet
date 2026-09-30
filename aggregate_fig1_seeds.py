"""Fig. 1 across seeds: held-out conditioning / smoothness of M1-M4 (L1 penalty).

Seed 0 files: figures/landscape/fig1/{eig_test_r*_M?.npz, hessian_test_M?.json}
Seed s > 0:   same names with suffix _s{s} (run_fig1_seeds.sh).

  kink ratio  s(2h)/s(h) of the held-out own loss along the top Hessian direction at the
              trained weights (mean of both sides): ~2 smooth quadratic minimum, ~1 V-shaped kink
  stiffness   lam_max / mean random-direction curvature of the held-out own loss
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_landscape_compare import build_losses, normalize

OUT = "figures/landscape/fig1"
MODELS = ["M1", "M2", "M3", "M4"]
TITLES = {"M1": r"M1: SL, small $\rho$", "M2": r"M2: SL, large $\rho$",
          "M3": r"M3: SL + FS, small $\rho$", "M4": r"M4: SSL, small $\rho$"}
SEEDS = [0, 1, 2]


def tag(m, s):
    return m if s == 0 else f"{m}_s{s}"


def own(m, s, R):
    f = f"{OUT}/eig_test_r{R}_{tag(m, s)}.npz"
    if not os.path.exists(f):
        return None
    d = dict(np.load(f, allow_pickle=True))
    L = build_losses(d, m, 10.0, 1e3, 1e5)
    c = {k: d[f"{m}/{k}"] for k in d["components"]}
    Z = {"M1": L["L_sl_l1_small"], "M2": L["L_sl_l1_high"],
         "M3": 100 * c["huber_fs"] + 10 * c["viol_l1"], "M4": L["L_ssl_l1_small"]}[m]
    return d["xs"], Z


def kink_ratio(xs, Z):
    c = len(xs) // 2
    line = Z[c, :]
    r = [((line[c + 2 * g] - line[c]) / 2) / (line[c + g] - line[c]) for g in (-1, 1)]
    return float(np.mean(r))


def main():
    rows = {m: {"kink": [], "stiff": [], "lam_sign": []} for m in MODELS}
    for m in MODELS:
        for s in SEEDS:
            g = own(m, s, "0.1")
            h = f"{OUT}/hessian_test_{tag(m, s)}.json"
            if g is None or not os.path.exists(h):
                continue
            rows[m]["kink"].append(kink_ratio(*g))
            hh = json.load(open(h))["own"]
            rows[m]["stiff"].append(hh["stiffness"])
            rows[m]["lam_sign"].append("min" if hh["c_rand"] > 0 else "saddle")

    fmt = lambda v: f"{np.mean(v):.3g} +- {np.std(v):.2g}" if v else "n/a"
    print("| model | seeds | kink ratio along top direction (2 = smooth, 1 = V) | held-out stiffness | curvature sign per seed |")
    print("|---|---:|---:|---:|---|")
    for m in MODELS:
        r = rows[m]
        print(f"| {m} | {len(r['kink'])} | {fmt(r['kink'])} ({', '.join(f'{v:.2f}' for v in r['kink'])}) | "
              f"{fmt(r['stiff'])} | {', '.join(r['lam_sign'])} |")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True, constrained_layout=True)
    for ax, m in zip(axes, MODELS):
        for s in SEEDS:
            g = own(m, s, "0.5")
            if g is None:
                continue
            xs, Z = g
            ax.plot(xs, normalize(Z)[len(xs) // 2, :] + 1e-3, label=f"seed {s}")
        ax.set_yscale("log"), ax.set_title(TITLES[m]), ax.set_xlabel("top Hessian direction")
    axes[0].set_ylabel("normalized held-out training loss"), axes[0].legend(fontsize=8)
    fig.savefig(f"{OUT}/fig1_eig_slices_seeds.png", dpi=200)
    print(f"\nFigure: {OUT}/fig1_eig_slices_seeds.png")


if __name__ == "__main__":
    main()
