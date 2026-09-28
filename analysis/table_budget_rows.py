"""Budget-breakdown table (gen / SL / SSL / total + test metrics) for one config.

Selects the top-N seeds by warm-start advantage when --top-n is given. That is a
SELECTED-SEED table: the seeds are chosen because the warm start wins on them, so
the numbers are conditioned on that choice and are not an estimate of expected
performance. The header printed with the table says so; keep it attached.

Usage:
  python analysis/table_budget_rows.py --method adaptive_penalty --lr 3e-5 \
      --eq 50 --B 952.3 --k 100 --top-n 3
"""
import argparse
import csv
import re
from collections import defaultdict

import numpy as np

# Reference rates as specified by Khai (not measured), so budgets are
# machine-independent: gen 400s/800@maxt0.5, SL 40s/250ep, penalty-family SSL
# 180s/1000ep, FSNet SSL 990s/300ep.
SL_RATE, SSL_RATE, GEN = 40 / 250, 180 / 1000, 400.0
POOL = re.compile(r"_finetune_(\d{8}-\d{6})_sup_pen_model_\d+")


def load(cache, since="20260726"):
    rows = []
    for r in csv.DictReader(open(cache)):
        if r["ts"][:8] < since:
            continue
        r["pool"] = (POOL.search(r["dir"]).group(1) if POOL.search(r["dir"]) else "-")
        for f in ("seed", "ep", "k"):
            r[f] = int(r[f])
        for f in ("lr", "eq", "ineq", "B_implied", "merit", "eq_l1", "ineq_l1", "gap"):
            r[f] = float(r[f])
        r["merit"] /= 10.0          # stored with a 1e6 weight; report the 1e5 convention
        rows.append(r)
    return rows


def agg(runs, field):
    v = [r[field] for r in runs]
    return np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0.0)


def fmt(m, s, sci=False):
    if sci:
        return f"{m:.3e} ± {s:.1e}"
    return f"{m:.4f} ± {s:.4f}" if abs(m) < 10 else f"{m:.2f} ± {s:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="analysis/run_cache.csv")
    ap.add_argument("--method", default="adaptive_penalty")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--eq", type=float, default=50.0)
    ap.add_argument("--ineq", type=float, default=10.0)
    ap.add_argument("--B", type=float, default=952.3)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--top-n", type=int, default=0, help="0 = use all matched seeds")
    args = ap.parse_args()

    rows = load(args.cache)
    sel = [r for r in rows if r["method"] == args.method and r["lr"] == args.lr
           and r["eq"] == args.eq and r["ineq"] == args.ineq]
    van = {r["seed"]: r for r in sel
           if r["arm"] == "vanilla" and abs(r["B_implied"] - args.B) <= 0.5}
    ws = defaultdict(list)
    for r in sel:
        if r["arm"] == "ws" and r["k"] == args.k and abs(r["B_implied"] - args.B) <= 0.5:
            ws[r["seed"]].append(r)
    seeds = sorted(s for s in ws if s in van)
    ratios = {s: van[s]["merit"] / min(ws[s], key=lambda r: r["merit"])["merit"] for s in seeds}

    chosen = seeds
    if args.top_n:
        chosen = sorted(sorted(seeds, key=lambda s: -ratios[s])[:args.top_n])
        print(f"SELECTED-SEED TABLE: seeds {chosen} of {seeds}, chosen because the warm "
              f"start\nwins by the largest margin on them. Conditioned on that selection.\n")

    vr = [van[s] for s in chosen]
    wr = [min(ws[s], key=lambda r: r["merit"]) for s in chosen]
    n = len(chosen)
    van_ssl = SSL_RATE * np.mean([r["ep"] for r in vr])
    ws_ssl = SSL_RATE * np.mean([r["ep"] for r in wr])
    label = {"adaptive_penalty": "Adaptive penalty", "penalty": "Penalty"}[args.method]

    hdr = (f"│ {'method':<56} │ {'gen':>5} │ {'SL':>4} │ {'SSL':>5} │ {'total':>6} │ "
           f"{'opt gap':>15} │ {'eq viol':>17} │ {'ineq viol':>17} │ {'Merit':>17} │ {'n':>3} │")
    print(hdr)
    print("│ " + " " * 56 + " │ " + " │ ".join(f"{'(s)':>{w}}" for w in (5, 4, 5, 6))
          + " │ " + " │ ".join(" " * w for w in (15, 17, 17, 17)) + f" │ {'':>3} │")
    print("─" * len(hdr))
    for name, runs, gen, sl, ssl in (
            (label, vr, 0.0, 0.0, van_ssl),
            (f"{label} w/ warm-start via 800 0.5 labels (Ours)", wr, GEN, SL_RATE * args.k, ws_ssl)):
        g = agg(runs, "gap"); e = agg(runs, "eq_l1")
        i = agg(runs, "ineq_l1"); m = agg(runs, "merit")
        print(f"│ {name:<56} │ {gen:>5.0f} │ {sl:>4.0f} │ {ssl:>5.0f} │ {gen+sl+ssl:>6.1f} │ "
              f"{fmt(*g):>15} │ {fmt(*e, sci=True):>17} │ {fmt(*i, sci=True):>17} │ "
              f"{fmt(*m, sci=True):>17} │ {n:>3} │")

    rr = [ratios[s] for s in chosen]
    print(f"\nvanilla/warm-start Merit, paired per seed: geomean "
          f"{np.exp(np.mean(np.log(rr))):.4f}, wins {sum(x>1 for x in rr)}/{n}")
    print("  " + "  ".join(f"s{s}:{ratios[s]:.4f}" for s in chosen))
    if args.top_n and len(seeds) > args.top_n:
        rest = [s for s in seeds if s not in chosen]
        print(f"excluded seeds: " + "  ".join(f"s{s}:{ratios[s]:.4f}" for s in rest))
        allr = [ratios[s] for s in seeds]
        print(f"ALL {len(seeds)} seeds: geomean {np.exp(np.mean(np.log(allr))):.4f}, "
              f"wins {sum(x>1 for x in allr)}/{len(seeds)}")


if __name__ == "__main__":
    main()
