"""Paired vanilla-vs-merit summary for the adaptive_penalty / penalty replication runs.

The generic analyze_budget_sweep.py keys runs on (seed, epochs, checkpoint), which
collides here: the same manifest row is run at several LRs and several
eq_pen_weights, so those runs are indistinguishable by that key. Directory names
DO encode lr / eq / ineq, so parse them instead.

Usage: python analysis/adapen_rep_summary.py [--method adaptive_penalty] [--since "2026-07-27 01:00"]
"""
import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np

RES = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"

PAT = re.compile(
    r"^(?P<ts>\d{8}-\d{6})_MLP_(?P<method>.+?)_seed(?P<seed>\d+)"
    r"_nepochs(?P<ep>\d+)_lr(?P<lr>[0-9.e+-]+)_trainsize(?P<ts_n>\d+)"
    r"_obj(?P<obj>[0-9.]+)_eq(?P<eq>[0-9.]+)_ineq(?P<ineq>[0-9.]+)"
    r"_lrsched(?P<sched>\w+?)_etamin(?P<etamin>[0-9.e+-]+)"
    r"(?:_finetune_(?P<fts>\d{8}-\d{6})_sup_pen_model_(?P<k>\d+))?$"
)


def scan(method, since_ts):
    runs = []
    with os.scandir(RES) as it:
        for e in it:
            if not e.is_dir() or f"_MLP_{method}_seed" not in e.name:
                continue
            m = PAT.match(e.name)
            if not m or m.group("method") != method:
                continue
            if m.group("ts") < since_ts:
                continue
            p = os.path.join(e.path, "results.pkl")
            if not os.path.exists(p):
                continue
            try:
                r = pickle.load(open(p, "rb"))
            except Exception:
                continue
            d = m.groupdict()
            mm = r["test_results"]["batch_size_comparison"][256]["metrics"]
            runs.append(dict(
                name=e.name, seed=int(d["seed"]), ep=int(d["ep"]), lr=float(d["lr"]),
                eq=float(d["eq"]), ineq=float(d["ineq"]),
                arm="merit" if d["k"] else "vanilla", k=int(d["k"] or 0),
                merit=float(mm["merit_mean"]),
                eq_l1=float(mm["eq_violation_l1_mean"]),
                ineq_l1=float(mm["ineq_violation_l1_mean"]),
                gap=float(mm["opt_gap_mean"]),
            ))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="adaptive_penalty")
    ap.add_argument("--since", default="20260727-000000")
    ap.add_argument("--seeds", default="", help="comma list to restrict to")
    args = ap.parse_args()

    runs = scan(args.method, args.since.replace(" ", "-").replace(":", "").replace("--", "-"))
    if args.seeds:
        keep = {int(s) for s in args.seeds.split(",")}
        runs = [r for r in runs if r["seed"] in keep]
    print(f"{len(runs)} runs, method={args.method}\n")

    # group by (lr, eq, ineq) config, pair vanilla vs merit within seed
    cfgs = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        cfgs[(r["lr"], r["eq"], r["ineq"])][r["seed"]][r["arm"]] = r

    hdr = f"{'lr':>8} {'eq':>7} {'ineq':>6} {'n':>3} {'van':>10} {'merit':>10} {'van/merit':>10} {'wins':>6}  per-seed"
    print(hdr)
    print("-" * len(hdr))
    for cfg in sorted(cfgs):
        seeds = cfgs[cfg]
        pairs = [(s, d["vanilla"], d["merit"]) for s, d in sorted(seeds.items())
                 if "vanilla" in d and "merit" in d]
        if not pairs:
            continue
        ratios = [v["merit"] / m["merit"] for _, v, m in pairs]
        wins = sum(x > 1 for x in ratios)
        van = np.mean([v["merit"] for _, v, _ in pairs])
        mer = np.mean([m["merit"] for _, _, m in pairs])
        gm = float(np.exp(np.mean(np.log(ratios))))
        ps = " ".join(f"s{s}:{x:.3f}" for (s, _, _), x in zip(pairs, ratios))
        print(f"{cfg[0]:>8.0e} {cfg[1]:>7.0f} {cfg[2]:>6.0f} {len(pairs):>3} "
              f"{van:>10.4g} {mer:>10.4g} {gm:>10.4f} {wins:>3}/{len(pairs)}  {ps}")

    print("\nper-run detail (eq_l1 / ineq_l1 / gap):")
    for cfg in sorted(cfgs):
        for s, d in sorted(cfgs[cfg].items()):
            for arm in ("vanilla", "merit"):
                r = d.get(arm)
                if r:
                    print(f"  lr={cfg[0]:.0e} eq={cfg[1]:.0f} s{s:<2} {arm:<8} k={r['k']:<4} "
                          f"ep={r['ep']:<5} merit={r['merit']:.4g} eq_l1={r['eq_l1']:.3g} "
                          f"ineq_l1={r['ineq_l1']:.3g} gap={r['gap']:.3g}")


if __name__ == "__main__":
    main()
