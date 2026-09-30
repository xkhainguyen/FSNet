# Handoff: ws-pen / ws-adapen budget study (2026-07-27)

Written for a fresh assistant (Cursor) picking this up because the Claude
session hit its weekly limit. Everything below is verified against the actual
run outputs on disk, not just prior chat summaries.

**Repo**: `/home/khain/FSNet` (== `/orcd/home/002/khain/FSNet`, same file, two
mounts — don't be confused if `pwd` shows either). Conda env `ml4opt`
(`source ~/.bashrc && conda activate ml4opt`).

## The task

We already showed **ws-adapen beats vanilla adapen** at equal total offline
budget (FSNet family — see prior sessions; solid result). We are now trying to
show **ws-pen beats vanilla pen** — a warm-start rule for the `penalty` /
`adaptive_penalty` methods on the SOCP problem — at the **same total offline
budget** `B = gen + SL + SSL`.

**Rate convention — use these exactly, do not re-measure from wall clock.**
Khai specified them so the study is machine-independent:

| quantity | rate |
|---|---|
| gen, 800 labels @ maxt0.5 | **400 s** flat |
| SL (`sup_pen`, 800 labeled instances) | **40 s / 250 epochs** = 0.16 s/ep |
| penalty / adaptive_penalty SSL | **180 s / 1000 epochs** = 0.18 s/ep |
| FSNet SSL (not this thread's focus) | 990 s / 300 epochs = 3.30 s/ep |

`SSL_epochs = round((B - gen - SL_rate*k) / SSL_rate)` for a warm-start arm
starting from SL checkpoint `k`; vanilla gets `SSL_epochs = round(B / SSL_rate)`
with no gen/SL cost (it uses no labels).

An earlier pass in this session used a *measured* gen cost of 396.3 s instead
of the specified 400 s flat. That has been corrected in the tooling (see below)
but a few already-completed runs (the k=200 AdaPen cell) were built off 396.3
and are ~3.6s (0.4%) short of nominal B under the 400s convention — noted
in-place where relevant below, not silently fixed.

## Bottom line so far

**No penalty-family configuration has a real, budget-matched, multi-seed win
for the warm-start arm.** Every apparent win found so far is either a small
selected-seed subset (2-4 of 8 seeds, picked because the warm start wins on
them) or a single seed. Over all 8 seeds, every arm in every family is a tie or
a loss (best geomean ≈ 1.00, i.e. flat).

**Why, mechanically:** a warm start is worth roughly the SSL time it takes
vanilla to reach the same validation `stop_merit` — measured at ~20-30s of SSL
for penalty (see `analysis/BUDGET_STUDY_NOTES.md` section "Session 3"). Label
generation for 800 instances alone costs 400s. So the warm start's overhead
(`gen + SL`) is ~15-20x larger than what it's worth, independent of B. This
holds for both `penalty` and `adaptive_penalty` — AdaPen's larger effect at
n=3-4 (see below) does not survive to n=8.

## What's running right now (as of last check, 2026-07-27 ~14:xx)

Two SLURM arrays, both k=250, both families, **all 8 seeds each**, budgets
matched to the fixed-rate convention:

| job | manifest | method | lr | eqW/ineqW | B (s) | epochs |
|---|---|---|---|---|---:|---:|
| `19003048` (adaK250) | `mf_adapen_k250_s0to7.tsv` | adaptive_penalty | 3e-5 | 50/10 (caps 500/100) | 952.4 | 2847 |
| `19003049` (penK250) | `mf_pen_k250_s0to7.tsv` | penalty | 1e-4 | 50/10 | 1100.0 | 3667 |

Vanilla already exists at these exact budgets for all 8 seeds in both families
(no new vanilla runs needed — see `analysis/run_cache.csv`, arm="vanilla").

**To check status:** `squeue -u khain` and `sacct -j 19003048,19003049 -X`.
**To pull results once done:** rebuild the cache (below), then query by
`k==250`.

## Tooling (all in `analysis/`)

- **`build_run_cache.py`** — one-pass scan of every penalty/adaptive_penalty
  run dir under
  `results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/`, parses lr /
  eq / ineq / seed / epochs / checkpoint straight out of the directory name
  (this is necessary — the generic `analyze_budget_sweep.py` keys on
  `(seed, epochs, checkpoint)` alone, which collides badly once you vary lr or
  penalty weights), pulls test metrics from `results.pkl`, and writes
  `analysis/run_cache.csv`. Re-run this after any new jobs finish:
  ```bash
  python analysis/build_run_cache.py
  ```
  Takes several minutes (reads ~300 pickles off the pool filesystem). Run it
  in the background and wait on the PID rather than polling.

- **`find_ws_wins.py`** — exhaustive scan of `run_cache.csv` for any
  budget-matched cell (method/lr/eq/ineq/schedule/seed/SL-pool matched,
  budgets within one SSL epoch) where a warm-start checkpoint beats vanilla.
  Key point: **each checkpoint k is its own arm** — do not collapse checkpoints
  into a single "merit" arm, that hides real cells (this mistake happened once
  already this session and had to be redone).
  ```bash
  python analysis/find_ws_wins.py --since 20260726 --min-n 2
  ```

- **`table_budget_rows.py`** — prints the gen/SL/SSL/total + opt-gap/eq-viol/
  ineq-viol/Merit table for one (method, lr, eq, ineq, B, k) config, optionally
  restricted to the top-N seeds by warm-start margin (`--top-n`, prints a
  selection-caveat header — keep it attached to any table pulled this way).
  ```bash
  python analysis/table_budget_rows.py --method adaptive_penalty --lr 3e-5 \
      --eq 50 --B 952.4 --k 250 --top-n 0   # 0 = all matched seeds
  ```
  Rates are hardcoded at the top of the file as
  `SL_RATE, SSL_RATE, GEN = 40/250, 180/1000, 400.0` — already fixed to the
  spec above.

- **`adapen_rep_summary.py`** — earlier/cruder version of the pairing logic in
  `find_ws_wins.py`, kept for reference; prefer the other two.

## Key results this session, checkpoint-resolved

Everything below is `Merit = obj + 1e5*(eq_l1 + ineq_l1)`, test split, batch
256, paired per-seed ratio = vanilla_merit / ws_merit (>1 means the warm start
wins).

### AdaPen, lr=3e-5, eqW=50/10, B=952.4 — by checkpoint k

| k | n | geomean (all avail. seeds) | wins |
|---:|---:|---:|---:|
| 100 | 8 | 0.9966 | 4/8 |
| 200 | 8 | 1.0000 | 4/8 (exact tie) |
| 250 | — | **running now, job 19003048** | |
| 350 | 4 (seeds 0-3 only) | 0.9832 | 1/4 |
| 500 | 4 (seeds 0-3 only) | 0.9926 | 1/4 |
| 650-850 | 1-2 seeds only | mixed, not reliable | |

On the best 3-4 seeds selected post hoc (2,3,4,6 or similar), k=200 gives
geomean ~1.02, but the seeds that "win" differ by checkpoint (s6 wins at every
k; s0/s1/s7 lose at every k tried) — the signature of seed noise, not a real
checkpoint effect. **All 8 seeds together: exactly 1.0000 at k=200, 0.9966 at
k=100.** Nothing beats vanilla on average at n=8 in this sweep.

### Penalty, lr=1e-4, eqW=50/10, B=1100 — by checkpoint

| arm | k | n | geomean | wins |
|---|---:|---:|---:|---:|
| early | 100 | 8 | 1.0028 | 3/8 (tie) |
| merit k* (rule's own pick) | 500-850, varies by seed | 8 | 0.9468 | 3/8 (loses) |
| conv | 950 | 8 | 0.9449 | 3/8 (loses) |
| 250 | — | **running now, job 19003049** | |

No data exists yet for penalty at k=200 or k=350 (only 100, k*, and 950 have
been run at n=8). k=250 currently running fills part of that gap.

### The one arm that isn't warm-starting at all

`early` (penalty, k=100) is a *tie* with vanilla and is the best-performing
warm-start arm found in the whole study — but it is NOT what the merit
selection rule picks (the rule picks k=500-850). So it can't be labeled "Ours"
if "Ours" means merit-based checkpoint selection.

### Important negative finding: LR sweep refutes the leading hypothesis

The working hypothesis was "the cosine LR restart at the start of SSL erases
the warm start." A symmetric LR sweep (both arms, same lr, penalty, B=1100)
refutes this directly:

| lr (both arms) | n | vanilla/merit | merit wins |
|---:|---:|---:|---:|
| 1e-4 | 8 | 0.9773 | 4/8 |
| 3e-5 | 4 | 0.8948 | 0/4 |
| 1e-5 | 2 | 0.9397 | 0/2 |

Lower LR makes the warm start *worse*, the opposite of the restart-erasure
prediction. It also makes vanilla much better in absolute terms, so lr=1e-4
(used in all the tables above) is the LR *most* favorable to the warm start,
not a neutral choice.

### The arithmetic that explains all of it

Warm-start value, measured directly (which vanilla epoch reaches the same
validation `stop_merit` as a given warm-start epoch): **~20-30s of SSL**, for
penalty at lr=1e-4. Generating 800 labels costs **400s** by the fixed
convention. Break-even requires `gen + SL <~ 27s`, i.e. roughly **50 labels**,
not 800. This is independent of B — no budget rescues it, because the
overhead-to-value ratio doesn't depend on B.

Separately: best achievable SL-checkpoint validation `stop_merit` at ANY label
quality tier (maxt 0.5 through 10.0) is **5.6x worse** than where vanilla
penalty (lr=1e-5) converges on its own. Better labels don't fix this — maxt2.0
labels are the *worst* SL init (800 instances is enough to overfit them).

## Files of interest

- `analysis/BUDGET_STUDY_NOTES.md` — the full running log, chronological,
  includes all the FSNet-family background (already-solid ws-adapen-vs-FSNet
  result) plus three "Session" sections on the penalty family. This is the
  most complete record; read it if `run_cache.csv` findings need context.
- `analysis/FINAL_TABLE_penalty_maxt05.md` — a specific selected/full seed
  table for penalty at B=1100, with caveats about selection and about the
  cosine-restart hypothesis (now updated to reflect the LR sweep refuting it).
- `analysis/run_cache.csv` — flat cache of every run (regenerate after new
  jobs finish; see `build_run_cache.py` above).
- `mf_adapen_k250_s0to7.tsv`, `mf_pen_k250_s0to7.tsv` — manifests for the
  currently-running jobs.
- `run_penalty_budget_pen05.sh` — the sbatch launcher both jobs above use.
  Reads `MANIFEST`, `METHOD`, `EQW`, `INEQW`, `EQMAX`, `INEQMAX` env vars;
  columns of the manifest are
  `idx  B  seed  q  arm_label  k  epochs  ckpt_path  lr`.

## Suggested next steps

1. Pull k=250 results once `19003048`/`19003049` drain (rebuild cache, query
   with `table_budget_rows.py --k 250`).
2. If k=250 doesn't beat vanilla at n=8 either (likely, given the pattern),
   the honest conclusion is that **the penalty family does not support a
   matched-budget warm-start win under these label/budget settings**, and the
   AdaPen/FSNet split from earlier sessions should be the reported positive
   result instead.
3. If a confirmed win is still wanted for the penalty family specifically,
   the only lever the arithmetic identifies is **far fewer labels** (~50, not
   800) — that's a different experiment (new SL pools at N=50, tiny B), not a
   checkpoint sweep at the current label count.
