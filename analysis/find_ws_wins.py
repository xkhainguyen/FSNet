"""Exhaustive search for any budget-matched cell where a warm-start arm beats vanilla.

Reads analysis/run_cache.csv. A warm-start run is paired with a vanilla run when
they share method / lr / eq / ineq / schedule / seed / SL pool AND their implied
total offline budgets agree to within one SSL epoch. Each distinct checkpoint k is
its own arm: collapsing all warm starts into a single "merit" arm is what hid the
E-sweep checkpoints in the earlier summary.

Ratio reported is vanilla_merit / ws_merit, so > 1 means the warm start wins.

Usage: python analysis/find_ws_wins.py [--since 20260726] [--min-n 2]
"""
import argparse
import csv
import re
from collections import defaultdict

import numpy as np

SSL_RATE = 0.18
POOL = re.compile(r"_finetune_(\d{8}-\d{6})_sup_pen_model_\d+")


def load(path, since):
    rows = []
    for r in csv.DictReader(open(path)):
        if r["ts"][:8] < since:
            continue
        m = POOL.search(r["dir"])
        r["pool"] = m.group(1) if m else "-"
        for f in ("seed", "ep", "k"):
            r[f] = int(r[f])
        for f in ("lr", "eq", "ineq", "B_implied", "merit"):
            r[f] = float(r[f])
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="analysis/run_cache.csv")
    ap.add_argument("--since", default="20260726")
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--tol", type=float, default=SSL_RATE * 2)
    args = ap.parse_args()

    rows = load(args.cache, args.since)
    van, ws = [r for r in rows if r["arm"] == "vanilla"], [r for r in rows if r["arm"] == "ws"]
    print(f"{len(rows)} runs since {args.since}: {len(van)} vanilla, {len(ws)} warm-start\n")

    # index vanilla by config+seed; several vanillas may exist at different budgets
    vidx = defaultdict(list)
    for r in van:
        vidx[(r["method"], r["lr"], r["eq"], r["ineq"], r["sched"], r["seed"])].append(r)

    cells = defaultdict(list)
    unmatched = defaultdict(int)
    for w in ws:
        key = (w["method"], w["lr"], w["eq"], w["ineq"], w["sched"], w["seed"])
        cands = [v for v in vidx.get(key, []) if abs(v["B_implied"] - w["B_implied"]) <= args.tol]
        if not cands:
            unmatched[(w["method"], w["lr"], w["eq"], w["ineq"], round(w["B_implied"]))] += 1
            continue
        v = min(cands, key=lambda v: abs(v["B_implied"] - w["B_implied"]))
        cell = (w["method"], w["lr"], w["eq"], w["ineq"], round(w["B_implied"]), w["pool"], w["k"])
        cells[cell].append((w["seed"], v["merit"], w["merit"]))

    out = []
    for cell, pairs in cells.items():
        pairs = sorted(set(pairs))
        rat = [v / m for _, v, m in pairs]
        out.append((float(np.exp(np.mean(np.log(rat)))), cell, pairs, rat))
    out.sort(reverse=True)

    hdr = (f"{'method':>18} {'lr':>7} {'eq':>6} {'ineq':>5} {'B':>6} {'k':>4} {'n':>3} "
           f"{'van/ws':>7} {'wins':>5}  per-seed")
    print(hdr)
    print("-" * len(hdr))
    for gm, cell, pairs, rat in out:
        meth, lr, eq, ineq, B, pool, k = cell
        if len(pairs) < args.min_n:
            continue
        wins = sum(x > 1 for x in rat)
        flag = "  <== WS WINS" if gm > 1 else ""
        ps = " ".join(f"s{s}:{x:.3f}" for (s, _, _), x in zip(pairs, rat))
        print(f"{meth:>18} {lr:>7.0e} {eq:>6.0f} {ineq:>5.0f} {B:>6.0f} {k:>4} {len(pairs):>3} "
              f"{gm:>7.4f} {wins:>2}/{len(pairs):<2}  {ps}{flag}")

    singles = [(gm, c, p, r) for gm, c, p, r in out if len(p) < args.min_n]
    if singles:
        print(f"\n{len(singles)} cells with n < {args.min_n} (not shown); "
              f"{sum(1 for gm, *_ in singles if gm > 1)} of them have ratio > 1")
    if unmatched:
        print(f"\nwarm-start runs with no budget-matched vanilla ({sum(unmatched.values())} runs):")
        for (meth, lr, eq, ineq, B), n in sorted(unmatched.items(), key=lambda t: -t[1])[:15]:
            print(f"  {meth:>18} lr={lr:.0e} eq={eq:.0f} ineq={ineq:.0f} B={B} : {n} runs")


if __name__ == "__main__":
    main()
