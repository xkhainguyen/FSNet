import os
"""Manifest for the penalty (not adaptive_penalty) fixed-budget test at maxt0.5.

Budget contract, charged with the reference rates regardless of which GPU the
job lands on:

    total = gen(800 labels @ maxt0.5) + SL_RATE * k + SSL_RATE * epochs

B is chosen so the warm-start arms land near 3000 SSL epochs, which is the scale
at which the adaptive_penalty warm start actually won. Vanilla spends the whole
budget on SSL and therefore gets ~2x the epochs; that is the point of the test,
not a handicap to be tuned away.
"""

SL_RATE = 0.16
SSL_RATE = 0.18
GEN05 = 396.3
B = 1100

# earliest k with val stop_merit <= 1.02 * min, per SL seed (maxt0.5 pools).
# Seeds 4-7 are a later replicate block; their k* was computed with the same
# rule before any SSL run, so it cannot have been tuned to the outcome.
KSTAR = {0: 750, 1: 800, 2: 500, 3: 700,
         4: 850, 5: 650, 6: 650, 7: 800}
SL_STAMP = {0: "20260726-125645", 1: "20260726-125645",
            2: "20260726-125645", 3: "20260726-125645",
            4: "20260726-184405", 5: "20260726-184402",
            6: "20260726-184708", 7: "20260726-184708"}
SL_DIR = ("results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/"
          "{t}_MLP_sup_pen_seed{s}_nepochs1000_lr0.0001_trainsize800_"
          "subopt_3_0.5_obj0.1_eq5.0_ineq5.0_lrschedcosine_etamin1e-06")

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3").split(",")]
OUT = os.environ.get("OUT", "budget_manifest_pen05.tsv")


def epochs(k, gen):
    return round((B - gen - SL_RATE * k) / SSL_RATE)


rows = []
idx = 0
for seed in SEEDS:
    d = SL_DIR.format(t=SL_STAMP[seed], s=seed)
    for arm, k in (("vanilla", 0), ("merit", KSTAR[seed]), ("early", 100), ("conv", 950)):
        gen = 0.0 if arm == "vanilla" else GEN05
        q = "-" if arm == "vanilla" else "0.5"
        ck = "-" if arm == "vanilla" else f"{d}/model_{k}.pt"
        rows.append((idx, B, seed, q, arm, k, epochs(k, gen), ck))
        idx += 1

with open(OUT, "w") as f:
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")

print(f"wrote {OUT}  ({len(rows)} rows, B={B}s)")
for r in rows:
    tot = (0.0 if r[4] == "vanilla" else GEN05) + SL_RATE * r[5] + SSL_RATE * r[6]
    print(f"  idx{r[0]:>3} seed{r[2]} {r[4]:>7} k={r[5]:>3} ep={r[6]:>5}  total={tot:7.1f}s "
          f"err={tot - B:+.2f}s")
