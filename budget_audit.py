"""Verify every arm in a manifest is charged the SAME total offline budget.

total = gen(regime, 800 labeled) + SL_RATE * k + SSL_RATE * epochs

Any arm whose total deviates from its nominal B by more than one SSL epoch of
rounding is a broken comparison and must not appear in a results table.

The vanilla arm has no gen and no SL by construction: it uses no labels, so its
entire budget goes to SSL. That is the intended asymmetry, not an error.

Usage: python budget_audit.py <manifest.tsv> [ssl_rate]
       ssl_rate defaults to 3.30 (FSNet); pass 0.18 for penalty/adaptive_penalty.
"""
import sys

SL_RATE = 0.16
GEN = {"0.5": 396.3, "1.5": 1171.5, "2.0": 1618.1, "-": 0.0}


def main():
    mf = sys.argv[1] if len(sys.argv) > 1 else "budget_manifest.tsv"
    ssl_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 3.30

    rows = []
    for line in open(mf):
        f = line.rstrip("\n").split("\t")
        if len(f) < 8:
            continue
        rows.append(dict(B=int(f[1]), seed=int(f[2]), q=f[3], arm=f[4],
                         k=int(f[5]), ep=int(f[6])))

    print(f"{mf}   ssl_rate={ssl_rate} s/ep   SL_RATE={SL_RATE} s/ep\n")
    hdr = (f"{'B':>5} {'seed':>4} {'regime':>7} {'arm':>8} {'k':>4} {'ep':>6} | "
           f"{'gen':>8} {'SL':>7} {'SSL':>8} {'total':>9} {'err':>7}")
    print(hdr)
    print("-" * len(hdr))
    worst = 0.0
    bad = 0
    for r in sorted(rows, key=lambda r: (r["B"], r["q"], r["arm"], r["seed"])):
        gen = GEN[r["q"]]
        sl = SL_RATE * r["k"]
        ssl = ssl_rate * r["ep"]
        tot = gen + sl + ssl
        err = tot - r["B"]
        worst = max(worst, abs(err))
        flag = ""
        if abs(err) > ssl_rate:          # more than one epoch of rounding
            flag = "  <-- MISMATCH"
            bad += 1
        print(f"{r['B']:>5} {r['seed']:>4} {r['q']:>7} {r['arm']:>8} {r['k']:>4} "
              f"{r['ep']:>6} | {gen:>8.1f} {sl:>7.1f} {ssl:>8.1f} {tot:>9.1f} "
              f"{err:>+7.1f}{flag}")

    print(f"\nrows={len(rows)}  max |err| = {worst:.2f}s  "
          f"(rounding tolerance = one SSL epoch = {ssl_rate:.2f}s)")
    print("VERDICT:", "all arms budget-matched" if bad == 0
          else f"{bad} arm(s) NOT budget-matched")

    # Per-arm share of the budget, which is what determines whether stopping
    # early can matter at all in a given regime.
    print("\nSL share of the post-gen remainder (merit can only help where this is large):")
    for B in sorted({r["B"] for r in rows}):
        for q in ("0.5", "2.0"):
            g = [r for r in rows if r["B"] == B and r["q"] == q]
            if not g:
                continue
            rem = B - GEN[q]
            conv = next((r for r in g if r["arm"] == "conv"), None)
            mer = [r for r in g if r["arm"] == "merit"]
            if not conv or not mer:
                continue
            km = sum(r["k"] for r in mer) / len(mer)
            print(f"  B={B} q={q}: remainder={rem:7.1f}s   "
                  f"conv SL={SL_RATE*conv['k']:6.1f}s ({100*SL_RATE*conv['k']/rem:4.1f}%)   "
                  f"merit SL={SL_RATE*km:6.1f}s ({100*SL_RATE*km/rem:4.1f}%)   "
                  f"epoch gain={100*(SL_RATE*(conv['k']-km))/(rem-SL_RATE*conv['k']):+5.1f}%")


if __name__ == "__main__":
    main()
