"""Summarize a penalty-family fixed-budget manifest with PAIRED per-seed ratios.

Runs are matched on (seed, num_epochs, checkpoint) read from results.pkl, not on
the directory name: the finetune tag records only the SL run's timestamp, and all
four seeds' SL pools share one timestamp, so paths are ambiguous.

Usage:
  python analyze_pen_budget.py <manifest.tsv> [--method=penalty] [--csv=out.csv]
"""
import glob
import os
import pickle
import sys

import numpy as np

RES = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
METRICS = ("merit_mean", "eq_violation_l1_mean", "ineq_violation_l1_mean",
           "objective", "opt_gap_mean")
SHORT = {"merit_mean": "merit", "eq_violation_l1_mean": "eq_l1",
         "ineq_violation_l1_mean": "ineq_l1", "objective": "obj",
         "opt_gap_mean": "opt_gap"}
ARM_ORDER = ("vanilla", "early", "merit", "conv", "merit3000")


def scan(method, want_eps):
    """Index candidate runs by (seed, num_epochs, checkpoint) from results.pkl.

    Pre-filter on the nepochs token in the directory name: there are hundreds of
    penalty runs on disk and unpickling all of them takes minutes.
    """
    tokens = tuple(f"_nepochs{e}_" for e in want_eps)
    idx = {}
    for d in glob.glob(f"{RES}/*_MLP_{method}_seed*"):
        if not any(t in os.path.basename(d) for t in tokens):
            continue
        p = os.path.join(d, "results.pkl")
        if not os.path.exists(p):
            continue
        try:
            r = pickle.load(open(p, "rb"))
        except Exception:
            continue
        cfg = r.get("config", {})
        ck = cfg.get("checkpoint") or "-"
        ne = cfg.get(cfg.get("method", method), {}).get("num_epochs")
        idx.setdefault((int(r.get("seed", -1)), ne, ck), []).append((d, r))
    return idx


def metrics_of(r):
    tr = r.get("test_results", {}).get("batch_size_comparison", {})
    if 256 in tr:
        return tr[256]["metrics"]
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    method = next((a.split("=", 1)[1] for a in sys.argv[1:]
                   if a.startswith("--method=")), "penalty")
    csv_out = next((a.split("=", 1)[1] for a in sys.argv[1:]
                    if a.startswith("--csv=")), None)
    mf = args[0] if args else "budget_manifest_pen05.tsv"

    spec = []
    for line in open(mf):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 8:
            spec.append(f[:8])
    idx = scan(method, {int(s[6]) for s in spec})

    rows = []
    for _, B, seed, q, arm, k, ep, ck in spec:
        hits = idx.get((int(seed), int(ep), ck), [])
        m = None
        if hits:
            d, r = sorted(hits)[-1]
            m = metrics_of(r)
        rows.append(dict(B=float(B), seed=int(seed), q=q, arm=arm, k=int(k),
                         ep=int(ep), m=m))

    done = [r for r in rows if r["m"]]
    print(f"{mf}  method={method}   rows={len(rows)}  finished={len(done)}  "
          f"missing={len(rows)-len(done)}\n")
    if not done:
        return

    if csv_out:
        cols = ["B", "seed", "q", "arm", "k", "ep"]
        with open(csv_out, "w") as f:
            f.write(",".join(cols + list(METRICS)) + "\n")
            for r in sorted(done, key=lambda r: (r["arm"], r["seed"])):
                f.write(",".join(str(r[c]) for c in cols) + "," +
                        ",".join(f"{r['m'].get(x, float('nan')):.6e}" for x in METRICS) + "\n")
        print(f"saved {len(done)} runs -> {csv_out}\n")

    hdr = f"{'arm':>10} {'q':>4} {'n':>2} {'ep':>6} " + " ".join(f"{SHORT[x]:>19}" for x in METRICS)
    print(hdr); print("-" * len(hdr))
    by = {}
    for r in done:
        by.setdefault(r["arm"], []).append(r)
    for arm in ARM_ORDER:
        g = by.get(arm)
        if not g:
            continue
        cells = []
        for x in METRICS:
            v = np.array([r["m"].get(x, np.nan) for r in g], float)
            cells.append(f"{np.nanmean(v):8.3e}+-{np.nanstd(v):8.2e}" if len(v) > 1
                         else f"{np.nanmean(v):8.3e}{'':10}")
        eps = sorted({r["ep"] for r in g})
        epstr = str(eps[0]) if len(eps) == 1 else f"{eps[0]}-{eps[-1]}"
        seeds = ",".join(str(r["seed"]) for r in sorted(g, key=lambda r: r["seed"]))
        print(f"{arm:>10} {g[0]['q']:>4} {len(g):>2} {epstr:>6} " + " ".join(cells)
              + f"  seeds={seeds}")

    # Paired per-seed ratios on merit_mean. Per-arm sigma here is 20-50%, so
    # comparing group means throws away nearly all the power; the arms share
    # seeds, and seed-level common variation cancels in the ratio.
    print("\npaired per-seed ratios on merit_mean (other/merit; >1 means merit better)")
    for ref in ("merit", "merit3000"):
        gm = {r["seed"]: r["m"]["merit_mean"] for r in by.get(ref, [])}
        if not gm:
            continue
        for other in ARM_ORDER:
            if other == ref or other not in by:
                continue
            go = {r["seed"]: r["m"]["merit_mean"] for r in by[other]}
            sh = sorted(set(gm) & set(go))
            if not sh:
                continue
            rr = np.array([go[s] / gm[s] for s in sh], float)
            wins = int((rr > 1).sum())
            verdict = (f"{ref} wins all" if wins == len(rr)
                       else f"{ref} loses all" if wins == 0 else f"split {wins}/{len(rr)}")
            per = " ".join(f"s{s}={go[s]/gm[s]:.2f}" for s in sh)
            print(f"  {ref:>9} vs {other:<10} n={len(rr)} "
                  f"geomean={np.exp(np.log(rr).mean()):.3f}x  [{verdict}]   {per}")


if __name__ == "__main__":
    main()
