"""Generate the FSNet fixed-offline-budget sweep manifest.

One line per SSL run. The sbatch array indexes into this file, which keeps the
per-seed k* bookkeeping out of bash.

Budget accounting uses the reference s/ep rates measured on the original
machine, NOT wall clock on whatever GPU the job lands on:
    SL        0.16 s/ep
    FSNet SSL 3.30 s/ep
    gen(800)  396.3s at maxt0.5 (low), 1618.1s at maxt2.0 (medium)
            (exact sums of solve_time_sec over the first 800 instances)

    SSL_epochs = round((B - gen - 0.16 * k) / 3.30)

k* is the argmin of validation stop_merit over each SL run's 20-checkpoint
pool, computed per seed (the selection rule is automatic and per-run).
"""
import os

SL_RATE = 0.16
SSL_RATE = 3.30
GEN = {0.5: 396.3, 2.0: 1618.1}

# k* = EARLIEST k with stop_merit <= 1.02 * min, per (regime, seed).
# Plain argmin proved unstable: on rerun it moved 150-200 epochs in the q=0.5
# regime because that merit curve is flat near its minimum (conv is within
# 0-12% of optimal, and identical for seed 2). The earliest-within-tolerance
# rule is stable under rerun and charges less SL time. In q=2.0 the minimum is
# sharp (conv is 2.4-3.3x worse) and this rule agrees with argmin.
KSTAR = {
    (0.5, 0): 750, (0.5, 1): 800, (0.5, 2): 500, (0.5, 3): 700,
    (2.0, 0): 200, (2.0, 1): 150, (2.0, 2): 200, (2.0, 3): 150,
}

# SL run directories (timestamps collide across regimes, so key on both)
SL_DIR = {
    (0.5, 0): "20260726-125645", (0.5, 1): "20260726-125645",
    (0.5, 2): "20260726-125645", (0.5, 3): "20260726-125645",
    (2.0, 0): "20260726-125720", (2.0, 1): "20260726-125720",
    (2.0, 2): "20260726-125720", (2.0, 3): "20260726-125720",
}

RES = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
SL_SUFFIX = "_nepochs1000_lr0.0001_trainsize800_subopt_3_{q}_obj0.1_eq5.0_ineq5.0_lrschedcosine_etamin1e-06"

CONV_K = 950   # last checkpoint in the pool
EARLY_K = 100


def sl_ckpt(q, seed, k):
    d = SL_DIR[(q, seed)] + "_MLP_sup_pen_seed" + str(seed) + SL_SUFFIX.format(q=q)
    return os.path.join(RES, d, f"model_{k}.pt")


def epochs(B, q, k):
    return round((B - GEN[q] - SL_RATE * k) / SSL_RATE)


def main():
    rows = []
    # B=2000 first: tighter budget, so the medium regime cannot wash out a bad
    # init and the effect should be largest. B=2800 as the looser reference.
    # All runs are constant-LR, so arms differing in epoch count no longer also
    # differ in LR-schedule shape.
    for B, seeds in ((2000, (0, 1, 2, 3)), (2800, (0, 1, 2, 3))):
        for seed in seeds:
            rows.append((B, seed, "-", "vanilla", 0, round(B / SSL_RATE), "-"))
            for q in (0.5, 2.0):
                for arm, k in (("merit", KSTAR[(q, seed)]),
                               ("early", EARLY_K),
                               ("conv", CONV_K)):
                    rows.append((B, seed, q, arm, k, epochs(B, q, k), sl_ckpt(q, seed, k)))

    missing = [r for r in rows if r[6] != "-" and not os.path.isfile(r[6])]
    if missing:
        for r in missing:
            print("MISSING CKPT:", r[6])
        raise SystemExit(f"{len(missing)} checkpoint(s) missing; aborting")

    with open("budget_manifest.tsv", "w") as f:
        for i, r in enumerate(rows):
            f.write("\t".join(str(x) for x in (i,) + r) + "\n")

    print(f"wrote budget_manifest.tsv with {len(rows)} runs")
    print(f"{'idx':>4} {'B':>5} {'seed':>4} {'q':>4} {'arm':>7} {'k':>4} {'ep':>5}")
    for r in rows:
        print(f"{rows.index(r):>4} {r[0]:>5} {r[1]:>4} {str(r[2]):>4} {r[3]:>7} {r[4]:>4} {r[5]:>5}")
    tot = sum(r[5] for r in rows)
    print(f"\ntotal SSL epochs = {tot}  (~{tot*3.0/3600:.1f} GPU-h at ~3.0 s/ep observed)")


if __name__ == "__main__":
    main()
