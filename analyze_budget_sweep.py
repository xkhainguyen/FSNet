"""Summarize the FSNet fixed-offline-budget sweep.

Reads a manifest, locates each run's results.pkl, and reports mean +/- std over
seeds per (B, regime, arm), plus merit-vs-{vanilla,early,conv} ratios.

Usage: python analyze_budget_sweep.py [manifest.tsv ...]
"""
import glob
import os
import pickle
import sys

import numpy as np

RES = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
METRICS = ("eq_violation_l1_mean", "ineq_violation_l1_mean", "opt_gap_mean", "merit_mean")
SHORT = {"eq_violation_l1_mean": "eq_l1", "ineq_violation_l1_mean": "ineq_l1",
         "opt_gap_mean": "opt_gap", "merit_mean": "merit"}


_CACHE = {}


def _scan(method, sched):
    """Index every finished run by (seed, num_epochs, checkpoint) read from the
    saved config. Matching on the dir name is NOT sufficient: the finetune tag
    records only the SL run's timestamp, and all four q=0.5 SL pools share one
    timestamp (they launched as a single array), so paired-seed and fixed-SL
    runs are indistinguishable by path. The checkpoint stored in results.pkl is
    authoritative.
    """
    key = (method, sched)
    if key in _CACHE:
        return _CACHE[key]
    suffix = "_constlr" if sched == "const" else "_lrschedcosine_etamin1e-06"
    idx = {}
    for d in glob.glob(f"{RES}/*_MLP_{method}_seed*{suffix}*"):
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
    _CACHE[key] = idx
    return idx


def find_run(seed, epochs, ckpt, sched="cosine", method="FSNet"):
    hits = _scan(method, sched).get((seed, epochs, ckpt), [])
    return sorted(hits, key=lambda t: t[0])[-1] if hits else None


def load(manifests):
    rows = []
    for mf in manifests:
        for line in open(mf):
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            _, B, seed, q, arm, k, ep, ckpt = f[:8]
            method = os.environ.get("SWEEP_METHOD", "FSNet")
            sched = os.environ.get("SWEEP_SCHED", "cosine")
            hit = find_run(int(seed), int(ep), ckpt, sched=sched, method=method)
            base = dict(B=int(B), seed=int(seed), q=q, arm=arm, k=int(k),
                        ep=int(ep), src=mf)
            if hit is None:
                rows.append(dict(base, m=None, dir=None))
                continue
            d, r = hit
            m = r["test_results"]["batch_size_comparison"][256]["metrics"]
            rows.append(dict(base, m=m, dir=os.path.basename(d)))
    return rows


def save_csv(rows, path):
    """Persist every finished run as one CSV line, so a later disk incident
    cannot cost us the numbers even if the .pkl files go away."""
    cols = ["B", "seed", "q", "arm", "k", "ep"] + list(METRICS)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: (r["B"], r["q"], r["arm"], r["seed"])):
            if not r["m"]:
                continue
            f.write(",".join(str(r[c]) for c in cols[:6]) + "," +
                    ",".join(f"{r['m'][x]:.6e}" for x in METRICS) + "\n")
    n = sum(1 for r in rows if r["m"])
    print(f"saved {n} finished runs -> {path}\n")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv_out = next((a.split("=", 1)[1] for a in sys.argv[1:]
                    if a.startswith("--csv=")), None)
    manifests = args or ["budget_manifest.tsv"]
    rows = load(manifests)
    if csv_out:
        save_csv(rows, csv_out)
    done = [r for r in rows if r["m"]]
    print(f"runs in manifest(s): {len(rows)}   with results: {len(done)}   "
          f"missing: {len(rows)-len(done)}\n")

    groups = {}
    for r in done:
        groups.setdefault((r["B"], r["q"], r["arm"]), []).append(r)

    for B in sorted({k[0] for k in groups}):
        print(f"===== B={B}s")
        hdr = f"{'regime':>7} {'arm':>8} {'n':>2} {'ep':>5} " + " ".join(
            f"{SHORT[x]:>20}" for x in METRICS)
        print(hdr); print("-" * len(hdr))
        for q in ("-", "0.5", "2.0"):
            for arm in ("vanilla", "merit", "early", "conv"):
                g = groups.get((B, q, arm))
                if not g:
                    continue
                cells = []
                for x in METRICS:
                    v = np.array([r["m"][x] for r in g], float)
                    cells.append(f"{v.mean():9.3e}+-{v.std():8.2e}" if len(v) > 1
                                 else f"{v.mean():9.3e}{'':10}")
                seeds = ",".join(str(r["seed"]) for r in sorted(g, key=lambda r: r["seed"]))
                print(f"{q:>7} {arm:>8} {len(g):>2} {g[0]['ep']:>5} " + " ".join(cells)
                      + f"   seeds={seeds}")
        # PAIRED per-seed ratios. The arms share seeds, so seed-level common
        # variation (which is 30-50% of the metric here) cancels in the ratio.
        # Comparing group means instead throws that power away and calls
        # everything noise.
        print()
        print("   paired per-seed merit ratios (other/merit; >1 means merit better)")
        for q in ("0.5", "2.0"):
            gm = {r["seed"]: r["m"]["merit_mean"] for r in groups.get((B, q, "merit"), [])}
            if not gm:
                continue
            for other in ("vanilla", "early", "conv"):
                go = {r["seed"]: r["m"]["merit_mean"]
                      for r in groups.get((B, "-" if other == "vanilla" else q, other), [])}
                shared = sorted(set(gm) & set(go))
                if not shared:
                    continue
                rr = np.array([go[s] / gm[s] for s in shared], float)
                wins = int((rr > 1).sum())
                verdict = ("merit wins all" if wins == len(rr)
                           else "merit loses all" if wins == 0
                           else f"split {wins}/{len(rr)}")
                per = " ".join(f"s{s}={go[s]/gm[s]:.2f}" for s in shared)
                print(f"     q={q} vs {other:8s} n={len(rr)} "
                      f"geomean={np.exp(np.log(rr).mean()):.2f}x  [{verdict}]   {per}")
        print()


if __name__ == "__main__":
    main()
