"""Fixed-SL design: ONE pretrained SL model, vary only the SSL seed.

Every warm-start arm initialises from the same seed-0 maxt0.5 pool, so SL-pool
variance is removed and the arm-to-arm comparison sees only SSL seed noise. The
paired ratios then measure the checkpoint choice alone.

The SL pool is seed 0 by index, fixed before any SSL run -- not chosen by which
pool produced the nicest downstream numbers. k* = 750 comes from the same
pre-registered rule as everywhere else (earliest checkpoint within 1.02x of the
minimum validation stop_merit).

Vanilla needs no checkpoint, so its 8 rows are the runs already produced by the
per-seed sweeps at the identical config (B=1100, 6111 epochs); they are emitted
here for analysis and do not need resubmitting. Submit array 8-31.
"""
import os

SL_RATE = 0.16
SSL_RATE = 0.18
GEN05 = 396.3
B = 1100

SL_SEED = 0
KSTAR = 750
SL_DIR = ("results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/"
          "20260726-125645_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize800_"
          "subopt_3_0.5_obj0.1_eq5.0_ineq5.0_lrschedcosine_etamin1e-06")
SSL_SEEDS = range(8)


def epochs(k, gen):
    return round((B - gen - SL_RATE * k) / SSL_RATE)


rows = []
idx = 0
for seed in SSL_SEEDS:                      # vanilla block first: idx 0-7
    rows.append((idx, B, seed, "-", "vanilla", 0, epochs(0, 0.0), "-"))
    idx += 1
for arm, k in (("merit", KSTAR), ("early", 100), ("conv", 950)):
    for seed in SSL_SEEDS:                  # warm-start block: idx 8-31
        rows.append((idx, B, seed, "0.5", arm, k, epochs(k, GEN05),
                     f"{SL_DIR}/model_{k}.pt"))
        idx += 1

out = os.environ.get("OUT", "budget_manifest_pen05_fixedsl.tsv")
with open(out, "w") as f:
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")

print(f"wrote {out}  ({len(rows)} rows, B={B}s, SL pool = seed{SL_SEED}, k*={KSTAR})")
seen = set()
for r in rows:
    if r[4] in seen:
        continue
    seen.add(r[4])
    tot = (0.0 if r[4] == "vanilla" else GEN05) + SL_RATE * r[5] + SSL_RATE * r[6]
    print(f"  {r[4]:>7} k={r[5]:>3} ep={r[6]:>5}  total={tot:7.1f}s err={tot - B:+.2f}s "
          f"(x8 SSL seeds)")
print("submit: sbatch --array=8-31%6  (idx 0-7 vanilla already exist)")
