"""AdaPen (or penalty) LR x eq-weight grid at fixed budget, paired by seed.

Scans only directories whose name already matches the epoch, lr and eq-weight
tokens in the manifests, then confirms against the saved config. Globbing every
run in the results pool and unpickling it takes minutes; this takes seconds.

Usage: python adapen_grid.py [--method=adaptive_penalty]
"""
import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

RES = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
# (manifest, eq_pen_weight)
GRID = [("budget_manifest_adapen_lrsweep.tsv", 10),
        ("budget_manifest_adapen_lrlow.tsv", 10),
        ("budget_manifest_adapen_eqw50.tsv", 50)]

merit = lambda m: m["objective"] + 1e5 * (m["eq_violation_l1_mean"]
                                          + m["ineq_violation_l1_mean"])


def load(mf, eqw, method, acc):
    spec = [l.rstrip("\n").split("\t") for l in open(mf)]
    eps = {f"_nepochs{r[6]}_" for r in spec}
    idx = {}
    for d in glob.glob(f"{RES}/*_MLP_{method}_seed*_eq{float(eqw)}_*"):
        b = os.path.basename(d)
        if not any(e in b for e in eps):
            continue
        p = os.path.join(d, "results.pkl")
        if not os.path.exists(p):
            continue
        try:
            r = pickle.load(open(p, "rb"))
        except Exception:
            continue
        c = r.get("config", {})
        m = c.get("method", method)
        idx.setdefault((int(r.get("seed", -1)), c.get(m, {}).get("num_epochs"),
                        c.get("checkpoint") or "-",
                        float(c.get(m, {}).get("lr", 0))), []).append(r)
    for _, B, s, q, arm, k, ep, ck, lr in spec:
        h = idx.get((int(s), int(ep), ck, float(lr)), [])
        if h:
            mm = h[-1]["test_results"]["batch_size_comparison"][256]["metrics"]
            acc[(eqw, float(lr))][arm][int(s)] = (merit(mm),
                                                  mm["eq_violation_l1_mean"])


def main():
    method = next((a.split("=", 1)[1] for a in sys.argv[1:]
                   if a.startswith("--method=")), "adaptive_penalty")
    acc = defaultdict(lambda: defaultdict(dict))
    for mf, eqw in GRID:
        if os.path.exists(mf):
            load(mf, eqw, method, acc)

    print(f"{method}, maxt0.5, 800 labels, B=952.3s   Merit = obj + 1e5*vio\n")
    hdr = (f"{'eqW':>4} {'lr':>8} | {'vanilla':>24} | {'merit':>24} | "
           f"{'van/merit':>9}  wins")
    print(hdr); print("-" * len(hdr))
    for eqw, lr in sorted(acc, key=lambda t: (t[0], -t[1])):
        g = acc[(eqw, lr)]
        cells = []
        for arm in ("vanilla", "merit"):
            v = np.array([x[0] for x in g.get(arm, {}).values()])
            e = np.array([x[1] for x in g.get(arm, {}).values()])
            cells.append(f"{v.mean():.3e} eq{e.mean():.3f} n{len(v)}"
                         if len(v) else f"{'-':>24}")
        sh = sorted(set(g.get("merit", {})) & set(g.get("vanilla", {})))
        rr = ""
        if sh:
            r = np.array([g["vanilla"][s][0] / g["merit"][s][0] for s in sh])
            rr = f"{np.exp(np.log(r).mean()):.4f}x  {int((r > 1).sum())}/{len(sh)}"
        print(f"{eqw:>4} {lr:>8.0e} | {cells[0]:>24} | {cells[1]:>24} | {rr}")


if __name__ == "__main__":
    main()
