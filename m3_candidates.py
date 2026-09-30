"""M3 candidates (rho 10 / 1 / 0.8): deployment quality after FS and FS effect on the training loss."""
import argparse
import numpy as np
from compute_landscape_compare import load_problem, load_model, eval_components
from fig1_common import ckpt, own_loss
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["M3", "M3r1", "M3r08"])
ap.add_argument("--nseeds", type=int, default=3)
ap.add_argument("--collapse", type=float, default=10.0, help="merit after FS (500 it) above this = collapsed")
args = ap.parse_args()
prob = load_problem()
summary = {}
X, Y = [t[:500].to(prob.device) for t in prob.test_dataset.tensors]
print("| model | seed | merit after FS (50 it) | merit after FS (500 it) | objective after FS (500) | FS effect on own loss |")
print("|---|---:|---:|---:|---:|---:|")
for m in args.models:
    for s in range(args.nseeds):
        try:
            net, _ = load_model(prob, ckpt(m, s))
        except FileNotFoundError:
            print(f"| {m} | {s} | missing | | | |"); continue
        r = {}
        for it in (50, 500):
            r[it] = eval_components(net, prob, X, Y, dict(val_tol=1e-9, memory=30, max_iter=it, scale=1000), 500)
        c = r[50]
        raw = dict(c, huber_fs=c["huber"])
        eff = abs(own_loss(m, c) - own_loss(m, raw)) / own_loss(m, raw)
        m500 = r[500]['obj_fs'] + 1e5 * r[500]['viol_l1_fs']
        summary.setdefault(m, []).append((m500, eff))
        print(f"| {m} | {s} | {r[50]['obj_fs'] + 1e5 * r[50]['viol_l1_fs']:.4g} | {m500:.4g} | "
              f"{r[500]['obj_fs']:.3f} | {eff:.1%} |", flush=True)

print("\n| model | seeds | collapsed (merit > %g) | merit of good seeds (median) | FS effect: good / collapsed (median) |" % args.collapse)
print("|---|---:|---:|---:|---|")
for m, v in summary.items():
    v = np.array(v)
    bad = v[:, 0] > args.collapse
    g = v[~bad]
    print(f"| {m} | {len(v)} | {bad.sum()} ({bad.mean():.0%}) | {np.median(g[:, 0]) if len(g) else float('nan'):.3g} | "
          f"{np.median(g[:, 1]) if len(g) else float('nan'):.1%} / {np.median(v[bad, 1]) if bad.any() else float('nan'):.1%} |")
