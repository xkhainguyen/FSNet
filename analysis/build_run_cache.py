"""One-pass cache of every penalty-family run into a flat CSV.

Loading results.pkl for ~500 runs off the pool filesystem takes minutes, so do it
once and query the CSV afterwards. Records the parsed directory name (which is
what distinguishes runs at the same seed/epochs but different lr or penalty
weights) alongside the test metrics and the implied offline budget.

Usage: python analysis/build_run_cache.py [--out analysis/run_cache.csv]
"""
import argparse
import csv
import os
import pickle
import re

RES = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"

PAT = re.compile(
    r"^(?P<ts>\d{8}-\d{6})_MLP_(?P<method>penalty|adaptive_penalty)_seed(?P<seed>\d+)"
    r"_nepochs(?P<ep>\d+)_lr(?P<lr>[0-9.e+-]+)_trainsize(?P<tsz>\d+)"
    r"_obj(?P<obj>[0-9.]+)_eq(?P<eq>[0-9.]+)_ineq(?P<ineq>[0-9.]+)"
    r"_lrsched(?P<sched>\w+?)_etamin(?P<etamin>[0-9.e+-]+)"
    r"(?:_finetune_(?P<fts>\d{8}-\d{6})_sup_pen_model_(?P<k>\d+))?"
    r"(?P<rest>.*)$"
)

# Reference rates as specified by Khai (not measured), machine-independent:
# gen 400s/800@maxt0.5, SL 40s/250ep, penalty-family SSL 180s/1000ep.
SL_RATE = 40 / 250
SSL_RATE = 180 / 1000
GEN_MAXT05_800 = 400.0

FIELDS = ["dir", "ts", "method", "seed", "ep", "lr", "eq", "ineq", "sched", "etamin",
          "arm", "k", "rest", "B_implied", "merit", "eq_l1", "ineq_l1", "obj", "gap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/run_cache.csv")
    args = ap.parse_args()

    rows, skipped = [], 0
    with os.scandir(RES) as it:
        for e in it:
            if not e.is_dir() or "_MLP_" not in e.name:
                continue
            m = PAT.match(e.name)
            if not m:
                continue
            p = os.path.join(e.path, "results.pkl")
            if not os.path.exists(p):
                skipped += 1
                continue
            try:
                r = pickle.load(open(p, "rb"))
                mm = r["test_results"]["batch_size_comparison"][256]["metrics"]
            except Exception:
                skipped += 1
                continue
            d = m.groupdict()
            k = int(d["k"] or 0)
            ep = int(d["ep"])
            warm = d["k"] is not None
            b = SSL_RATE * ep + (GEN_MAXT05_800 + SL_RATE * k if warm else 0.0)
            rows.append({
                "dir": e.name, "ts": d["ts"], "method": d["method"], "seed": int(d["seed"]),
                "ep": ep, "lr": float(d["lr"]), "eq": float(d["eq"]), "ineq": float(d["ineq"]),
                "sched": d["sched"], "etamin": d["etamin"],
                "arm": "ws" if warm else "vanilla", "k": k, "rest": d["rest"],
                "B_implied": round(b, 1),
                "merit": mm["merit_mean"], "eq_l1": mm["eq_violation_l1_mean"],
                "ineq_l1": mm["ineq_violation_l1_mean"],
                "obj": mm.get("objective"), "gap": mm["opt_gap_mean"],
            })

    rows.sort(key=lambda r: (r["method"], r["ts"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} runs to {args.out} ({skipped} dirs had no readable results.pkl)")


if __name__ == "__main__":
    main()
