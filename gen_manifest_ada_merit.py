"""Merit-vs-early swap under the winning adaptive_penalty recipe (lr=5e-4,
cosine to eta_min=1e-6, ~3000 WS epochs) at maxt0.5 / 800 labels.

The recorded win used SL pool 20260116-022657, but that pool only kept ckpts
50..400 while merit on it selects k=660 -- the merit arm was not testable there.
It also predates the eta_min fix. We therefore use the July pools, which are the
same SL config (sup_pen, 800 labels, maxt0.5, 1000 ep, lr 1e-4) with the full
50..950 ckpt grid and the corrected cosine schedule.

Two arms are emitted per warm start:

  <arm>      epochs set so gen + SL + SSL equals B for every arm. This is the
             budget-fair comparison; later checkpoints buy fewer SSL epochs.
  merit3000  merit init at early's 3000 epochs. Deliberately OVER budget, so it
             isolates "is the merit checkpoint a better init" from "does merit
             cost more SL time". If merit loses here it loses on init quality,
             not on accounting.

B is pinned to the recorded win: early(k=100) at 3000 epochs.
"""

SL_RATE = 0.16
SSL_RATE = 0.18
GEN05 = 396.3
B = round(GEN05 + SL_RATE * 100 + SSL_RATE * 3000, 1)   # 952.3s

KSTAR = {0: 750, 1: 800, 2: 500, 3: 700}
SL_DIR = ("results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/"
          "20260726-125645_MLP_sup_pen_seed{s}_nepochs1000_lr0.0001_trainsize800_"
          "subopt_3_0.5_obj0.1_eq5.0_ineq5.0_lrschedcosine_etamin1e-06")


def epochs(k, gen):
    return round((B - gen - SL_RATE * k) / SSL_RATE)


rows = []
idx = 0
for seed in range(4):
    km = KSTAR[seed]
    arms = [("vanilla", 0, epochs(0, 0.0)),
            ("merit", km, epochs(km, GEN05)),
            ("early", 100, epochs(100, GEN05)),
            ("conv", 950, epochs(950, GEN05)),
            ("merit3000", km, 3000)]
    for arm, k, ep in arms:
        gen = 0.0 if arm == "vanilla" else GEN05
        q = "-" if arm == "vanilla" else "0.5"
        ck = "-" if arm == "vanilla" else f"{SL_DIR.format(s=seed)}/model_{k}.pt"
        rows.append((idx, B, seed, q, arm, k, ep, ck))
        idx += 1

with open("budget_manifest_ada_merit.tsv", "w") as f:
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")

print(f"wrote budget_manifest_ada_merit.tsv  ({len(rows)} rows, B={B}s)")
for r in rows:
    tot = (0.0 if r[4] == "vanilla" else GEN05) + SL_RATE * r[5] + SSL_RATE * r[6]
    note = "  OVER BUDGET (equal-epoch ablation)" if r[4] == "merit3000" else ""
    print(f"  idx{r[0]:>3} seed{r[2]} {r[4]:>9} k={r[5]:>3} ep={r[6]:>5}  "
          f"total={tot:7.1f}s err={tot - B:+7.1f}s{note}")
