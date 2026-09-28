"""How much does merit-based checkpoint selection buy, as a function of label
inexactness?  SL-side only -- no SSL runs required.

For each SL pool (one label-quality tier x one seed) we compare, on the
checkpointed grid:

  k_loss  = argmin of the SL training loss on the cheap (inexact) labels
            -- what you pick if you trust the supervised objective
  k_merit = earliest k with task merit <= 1.02 * min
            -- what you pick with a task-faithful signal

and report  merit(k_loss) / merit(k_merit).  A ratio of 1.0 means the
supervised objective happened to pick a task-optimal checkpoint; large ratios
mean following SL loss actively costs task performance.

Expectation (inverted U): near-zero cost when labels are so infeasible the model
cannot fit them, largest cost at intermediate inexactness, and small again when
labels are near-exact so that fitting them aligns with the task.
"""
import glob
import os
import pickle
import re

import numpy as np

RES = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
TOL = 1.02


def label_quality():
    """Infeasibility of the labels themselves over the 800 SL training instances."""
    base = ("datasets/nonsmooth_nonconvex/socp/"
            "random2025_socp_dataset_var100_ineq50_eq50_ex10000_maxt{}_ready")
    out = {}
    for f in glob.glob(base.format("*")):
        m = re.search(r"maxt([0-9.]+)_ready", f)
        if not m:
            continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        if "eq_l2" not in d:
            continue
        out[float(m.group(1))] = dict(
            eq=float(np.asarray(d["eq_l2"], float)[:800].mean()),
            ineq=float(np.asarray(d["ineq_max"], float)[:800].mean()),
            succ=100.0 * float(np.asarray(d["success"], bool)[:800].mean()),
        )
    return out


def pools():
    rows = []
    for d in glob.glob(f"{RES}/*sup_pen*trainsize800*etamin1e-06"):
        if len(glob.glob(os.path.join(d, "model_*.pt"))) < 20:
            continue  # pool pruned; cannot reproduce the arms
        m = re.search(r"seed(\d).*subopt_3_([0-9.]+)", d)
        if not m:
            continue
        try:
            r = pickle.load(open(os.path.join(d, "results.pkl"), "rb"))
        except Exception:
            continue
        vh, th = r["val_history"], {e["epoch"]: e["loss"] for e in r["train_history"]}
        ks = [e["epoch"] for e in vh]
        sm = np.array([e["stop_merit"] for e in vh], float)
        loss = np.array([th.get(k, np.nan) for k in ks], float)
        if np.all(np.isnan(loss[1:])):
            continue
        k_loss = ks[int(np.nanargmin(loss[1:])) + 1]          # skip ep 0
        k_merit = int(np.array(ks)[sm <= sm.min() * TOL].min())
        rows.append(dict(q=float(m.group(2)), seed=int(m.group(1)),
                         k_loss=k_loss, k_merit=k_merit,
                         ratio=sm[ks.index(k_loss)] / sm[ks.index(k_merit)],
                         merit_min=sm.min(),
                         flat=int((sm <= sm.min() * TOL).sum())))
    return rows


def main():
    lq = label_quality()
    rows = pools()
    if not rows:
        print("no complete SL pools found")
        return

    by_q = {}
    for r in rows:
        by_q.setdefault(r["q"], []).append(r)

    print("Value of task-faithful (merit) selection vs supervised-loss selection")
    print("ratio = merit(SL-loss pick) / merit(merit pick);  >1 means merit selection wins\n")
    hdr = (f"{'maxt':>5} {'n':>2} | {'label eq_l2':>11} {'label ineq':>10} {'succ%':>6} | "
           f"{'k_loss':>7} {'k_merit':>8} {'flat pts':>8} | {'ratio':>14}")
    print(hdr)
    print("-" * len(hdr))
    for q in sorted(by_q):
        g = by_q[q]
        rr = np.array([x["ratio"] for x in g], float)
        km = np.array([x["k_merit"] for x in g], float)
        kl = np.array([x["k_loss"] for x in g], float)
        fl = np.array([x["flat"] for x in g], float)
        L = lq.get(q, {})
        print(f"{q:>5} {len(g):>2} | {L.get('eq', float('nan')):>11.2e} "
              f"{L.get('ineq', float('nan')):>10.2e} {L.get('succ', float('nan')):>5.1f}% | "
              f"{kl.mean():>7.0f} {km.mean():>8.0f} {fl.mean():>8.1f} | "
              f"{rr.mean():>7.2f}x +-{rr.std():>5.2f}")
    print("\nper-seed detail")
    for q in sorted(by_q):
        for r in sorted(by_q[q], key=lambda x: x["seed"]):
            print(f"  maxt{q} seed{r['seed']}: k_loss={r['k_loss']:>4} "
                  f"k_merit={r['k_merit']:>4} ratio={r['ratio']:.2f}x "
                  f"(flat pts within {TOL:.2f}x: {r['flat']})")


if __name__ == "__main__":
    main()
