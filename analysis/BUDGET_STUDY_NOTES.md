# Fixed-offline-budget warm-start study — working notes

Session date: 2026-07-26. Status: **STOPPED** — all jobs cancelled at Khai's
request. Numbers below are final for what completed; gaps are marked.

**adaptive_penalty at B=1850 produced NO data** (cancelled while queued), so the
penalty family still has zero usable warm-start results. Its one predicted-testable
cell (merit 1111 ep vs conv 444 ep at q=2.0) is UNTESTED.

> **REPORTED CONFIG (decided): FSNet, cosine LR with eta_min=1e-6, per-arm
> T_max = that arm's epoch budget, robust k*, B=2800.** The constant-LR sweeps are
> superseded and kept only for the record.
>
> Why cosine, stated so the choice is outcome-independent: under constant LR a
> run never anneals, so its final checkpoint is on average **1.42x worse than the
> best model the run already passed through** (median over 26 runs; worst 5.3x),
> and the best model lands mid-run (median 58% of budget). That reported number is
> therefore not the model you would deploy. Under cosine final/best is 1.09x, i.e.
> the final model *is* the converged one. This is a measurement-quality argument
> that holds regardless of which hypothesis either schedule happened to favour.
>
> Honesty requirement: const LR and cosine give **opposite verdicts** on the same
> arms (const: merit>conv t=7.7, ties vanilla/early; cosine: merit>vanilla 2.32x
> and >early 1.47x on 3/3, conv split). State the schedule prominently in any
> table; do not present either as schedule-independent.

Goal: show that merit-based termination of SL pretraining is adaptive across
label-quality regimes and transfers, whereas a fixed checkpoint choice (very
early, or converged) does not — at equal total offline budget
`B = gen + SL + SSL`, and with better final SSL performance.

---

## 1. Budget accounting (the contract)

Epoch counts are derived from **reference s/ep rates**, never from wall clock on
whichever GPU a job lands on. This is deliberate so all arms/regimes stay
comparable across machines. Do not recompute epochs from H200 / RTX PRO 6000
timings.

| quantity | rate |
|---|---|
| SL (`sup_pen`, 800 labeled) | 0.16 s/ep |
| FSNet SSL | 3.30 s/ep |
| penalty / adaptive_penalty SSL | 0.18 s/ep |

```
SSL_epochs = round((B - gen - 0.16 * k) / ssl_rate)
```

### Measured gen cost (exact, not extrapolated)

Summed `solve_time_sec` over the first N instances of each `maxt{q}_ready`
dataset. These datasets record per-instance solve time, so gen cost is exact.

| maxt (quality) | mean s/inst | gen(800) | gen(7000) |
|---|---|---|---|
| 0.5 (low) | 0.496 | **396.3s** | 3467s |
| 1.0 | 0.960 | 766.1s | 6724s |
| 1.5 | 1.471 | 1171.5s | 10264s |
| 2.0 (medium) | 2.027 | **1618.1s** | 14182s |
| 3.0 | 3.023 | 2421.8s | 21167s |
| 4.0 | 3.785 | 3104.9s | 26620s |
| 10.0 / 0.0 | 4.017 | 3249.0s | 28344s |

Sanity check vs. prior figures: maxt10 × 7000 = 28344s vs 28118s reported; maxt0.5
× 800 = 396s vs 400s reported. Both match.

### Why B has a hard floor

`B >= gen(medium)`. With medium = maxt2.0 at 800 instances that is 1618s, so
**B = 1000 is infeasible** — the labels alone cost more than the budget. B=2000
and B=2800 were used.

### Structural consequence (important)

A *shared* B must satisfy `B >= gen(q=2.0)`, and `gen(q=2.0) ~ 4x gen(q=0.5)`.
So the low regime always has a large post-gen remainder, and SL cost stays a
negligible share of it. Verified: merit's budget advantage at q=0.5 is stuck at
**1.05–1.06x for N = 800, 2000, and 4000 labeled instances alike**. Scaling the
labeled set does not fix this; only a per-regime budget would.

Merit can beat conv by exactly two mechanisms, and at q=0.5 both are absent:

| mechanism | q=0.5 | q=2.0 |
|---|---|---|
| better init (sharp SL merit minimum) | absent — curve flat, conv within 0–12% | present — conv 2.4–3.3x worse |
| budget saved by early stop | absent — SL ~6% of remainder | present — conv's SL eats 40% |

---

## 2. Vehicle selection: FSNet works, plain penalty does not

`penalty` (fixed `eq_pen_weight: 10.0`) plateaus at `eq_l1 ~ 0.80` by epoch ~700
and flatlines. Verified two ways: our 11111-epoch run gives `eq_l1 = 0.7764`,
and an independent 2667-epoch run gives `0.8022` — statistically identical. So
epochs past ~1000 buy nothing, every arm sits on the same infeasible fixed
point, and vanilla was the *best* of 7 arms. **All conclusions from the first
penalty screen are void** (that includes an apparent q=0.5 merit failure, which
was plateau noise).

At `eq_l1 ~ 0.8` the negative opt gaps (-4 to -9.5) are *produced by* the
constraint violation, so those numbers do not measure solution quality.

FSNet by contrast reaches `eq_l1 ~ 1e-5`, `ineq_l1 ~ 5e-7`, and does not
saturate within budget (312–848 epochs). Its negative opt gaps are different in
kind: with violations at 1e-5 it genuinely beats the IPOPT reference, which is
itself a truncated local solve.

> Writeup note: define the reference explicitly as a truncated local solve, and
> consider reporting gap vs. best-known-per-instance. A negative optimality gap
> will otherwise trip reviewers.

`adaptive_penalty` reaches `eq_l1 = 0.645`, `ineq_l1 = 8.8e-4` — better than
plain penalty but still far from FSNet. Screen running (job 18921377).

**Prediction on record:** at 0.18 s/ep, any feasible B forces >= 1277 SSL
epochs, well past plain penalty's ~700-epoch plateau. If adaptive_penalty is
also saturated there, the whole penalty family is structurally unusable for a
budget study at this gen cost.

### 2b. Confirmed: warm-starting needs SSL to be the binding constraint

Budgets may differ per method family (they are different families; comparisons
only ever happen *within* a family). But choosing B for the penalty family
exposes a structural precondition.

`adaptive_penalty` saturates by **~490 epochs** (first within 5% of its best
merit; eq_l1 bottoms ~0.67-0.69 near epoch 440 then oscillates). Plain penalty
saturates ~1550, and at a worse value (0.86).

Which B keeps the conv arm below saturation? (\* = past ~490 ep, converged)

| B | q2.0 merit | q2.0 conv | q0.5 merit | q0.5 conv |
|---|---|---|---|---|
| 1800 | 833\* | **166** | 7621\* | 6954\* |
| 1850 | 1111\* | **444** | 7898\* | 7232\* |
| 1900 | 1388\* | 722\* | 8176\* | 7509\* |
| 2000 | 1944\* | 1277\* | 8732\* | 8065\* |

Medium works at B ~ 1800-1850. **The low regime never does**: any B feasible for
medium (>= 1770) gives the low regime 7000-8000 epochs, ~15x past saturation. It
would need its own budget of ~600.

**Root cause:** penalty SSL is 0.18 s/ep vs FSNet's 3.30 — **18x cheaper** — so
the same wall clock buys 18x more epochs and the SSL budget stops being the
binding constraint. Then:

- **penalty**: vanilla spends the whole budget on SSL, saturates, and reaches the
  method's asymptote. WS arms divert budget to gen+SL, get fewer epochs, and
  cannot beat an asymptote vanilla already reached -> **vanilla wins**. This, not
  the plateau alone, is why the plain-penalty screen had vanilla best.
- **FSNet**: 848 epochs at B=2800, far from saturation, so budget spent on labels
  buys a genuinely better starting point.

**Statable precondition:** warm-starting — and hence merit-based selection of the
warm start — pays only when SSL cost dominates label-generation cost. This is a
result, not a defect, and it fits the inverted-U framing: the method matters
exactly where there is something to gain.

Plan: FSNet is the primary vehicle. Run adaptive_penalty at **B=1850, medium
regime only, 4 seeds** (~12 runs, ~1 GPU-h) as a documented boundary case rather
than a two-regime sweep that would produce a table of ties. The B=2000 screen was
cancelled before running for this reason.

Caveat: the 490-epoch saturation figure comes from one prior run at lr=1e-4;
treat as approximate. One dense-eval 2000-epoch run would confirm it (~6 min).

---

## 3. k* selection: argmin is unstable, use earliest-within-tolerance

Original rule: `k* = argmin` of SL validation `stop_merit` over the 20-checkpoint
pool. On an exact rerun (same seed, same data, same code) it **moved 150–200
epochs in the low regime** while staying pinned in the medium regime:

| regime | run 1 k* | run 2 k* |
|---|---|---|
| q=0.5 | 900, 750, 850, 550 | 750, 900, 950, 700 |
| q=2.0 | 200, 150, 200, 200 | 200, 150, 200, 200 |

Cause is curve flatness, not the estimator. Merit values also shifted slightly
(2.3439e5 -> 2.3304e5), i.e. runs are not bit-identical — plausibly `fused=True`
in the AdamW construction (`utils/trainer.py:697-702`).

### Flatness of the SL validation merit curve

| regime | seed | k within 2% of min | merit(950)/min |
|---|---|---|---|
| 0.5 | 0 | 750, 900 | 1.08 |
| 0.5 | 1 | 800, 900 | 1.11 |
| 0.5 | 2 | 500, 750, 850, 950 | **1.00** |
| 0.5 | 3 | 700 | 1.12 |
| 2.0 | 0 | 200, 250 | **2.42** |
| 2.0 | 1 | 150, 200, 250 | **3.30** |
| 2.0 | 2 | 200 | 2.69 |
| 2.0 | 3 | 150, 200 | 2.46 |

**Mechanism (this is the paper's story):** high-quality labels induce
overfitting, which creates a sharp interior merit minimum that a stopping rule
can find and a fixed choice misses. Low-quality labels are too noisy to
overfit, so the curve is flat, conv is already near-optimal, and no rule can
gain. This predicts *when* the method helps.

### Adopted rule

`k* = earliest k with stop_merit <= 1.02 * min`. Stable under rerun, charges
less SL time, and agrees with argmin wherever the minimum is sharp.

| regime | argmin | robust k* |
|---|---|---|
| q=0.5 | 750, 900, 950, 700 | 750, 800, **500**, 700 |
| q=2.0 | 200, 150, 200, 150 | 200, 150, 200, 150 |

---

## 4. Results so far — FSNet, const LR, paired seeds

B=2000, n=2 seeds (0,1). Test split, batch 256. Lower merit is better.

| regime | arm | ep | eq_l1 | opt_gap | merit |
|---|---|---|---|---|---|
| — | vanilla | 606 | 8.04e-05 ± 4.3e-06 | -0.72 ± 8.0 | 80.2 ± 7.3 |
| 0.5 | merit | 450 | 5.24e-05 ± 2.4e-05 | -8.40 ± 0.34 | 49.5 ± 23.7 |
| 0.5 | early | 481 | 9.96e-05 ± 2.3e-05 | -0.55 ± 3.9 | 98.9 ± 21.1 |
| 0.5 | conv | 440 | 6.36e-05 ± 2.7e-05 | -8.44 ± 0.33 | 60.8 ± 27.1 |
| 2.0 | merit | 106 | 7.33e-05 ± 5.7e-06 | -2.00 ± 1.7 | 72.4 ± 6.7 |
| 2.0 | early | 111 | 6.03e-05 ± 2.6e-05 | 1.37 ± 5.0 | 60.8 ± 28.0 |
| 2.0 | conv | 70 | 8.30e-05 ± 3.3e-06 | -0.14 ± 0.11 | 83.0 ± 3.2 |

### Use PAIRED statistics, not group means

Per-arm sigma is **30–50% of the metric**, so group means declare nearly
everything noise. Arms share seeds, so per-seed ratios cancel the seed-level
common variation. This is the correct statistic for this design.

```
q=0.5 vs conv     geomean 1.25x  [merit wins all]   s0=1.31 s1=1.20
q=0.5 vs early    geomean 2.22x  [merit wins all]   s0=4.65 s1=1.06
q=0.5 vs vanilla  geomean 1.84x  [split 1/2]        s0=3.39 s1=1.00
q=2.0 vs conv     geomean 1.15x  [merit wins all]   s0=1.09 s1=1.21
q=2.0 vs vanilla  geomean 1.11x  [merit wins all]   s0=1.11 s1=1.11
q=2.0 vs early    geomean 0.75x  [split 1/2]        s0=1.12 s1=0.50
```

**Holding up at n=2:** merit beats conv in both regimes, 4/4 seed-arm wins,
1.09–1.31x. **Not holding:** merit vs early flips at q=2.0 seed 1 (early 32.9 vs
merit 65.7); merit vs vanilla ties at q=0.5 seed 1. So "merit beats all three"
is still open, and `early` is the problem child — not `conv`, which was the
original suspicion.

### Superseded / preliminary

An earlier **cosine** seed-0 FSNet table had merit *losing* to conv at q=0.5
(0.63x). Two things changed since: const LR, and robust k*=750 vs argmin 900.
Not separable yet — a cheap disentangling run would be argmin k*=900 under
const LR. Treat the cosine table as preliminary; do not put it in the paper.

### Budget sweet spot

B=2000 may over-tighten. Its q=2.0 arms run 70–111 epochs and all land at merit
79–89, barely distinguishable from vanilla's 87.5. At B=2800 the same arms had
312–353 epochs and separated cleanly (merit 25.9 vs conv 36.3, 1.40x). Starving
SSL compresses everything toward undertrained rather than sharpening the
comparison. B=2800 half is running to bracket this.

---

## 4b. WHY merit: task-faithful vs supervised-loss selection

Thesis: merit is task-faithful (objective + constraint violation, computable
**without labels**), whereas SL loss fits inexact labels and so overfits them.
Therefore the merit-selected checkpoint should be the best SSL warm start — at
equal total offline budget.

`sl_selection_value.py` tests the SL half of this with **no SSL compute**. For
each pool it compares the checkpoint chosen by SL loss against the one chosen by
task merit:

| maxt | label eq_l2 | label ineq_max | k_loss | k_merit | merit(loss pick)/merit(merit pick) |
|---|---|---|---|---|---|
| 0.5 | 3.79 | 32.2 | 925 | 688 | 1.07x ± 0.05 |
| 1.0 | 2.92 | 24.4 | 950 | 675 | 1.09x ± 0.09 |
| 1.5 | 1.33 | 12.1 | 950 | **225** | 1.74x ± 0.35 |
| 2.0 | 0.403 | 4.80 | 950 | **175** | **2.71x ± 0.36** |
| 3.0 / 4.0 / 10.0 | 0.017 -> 1.5e-14 | | — | — | pools queued (18924184) |

Monotone rise 1.07 -> 1.09 -> 1.74 -> 2.71 with tight error bars.

**Mechanistic signature:** `k_merit` sits LATE (688, 675) while labels are grossly
infeasible, then collapses to EARLY (225, 175) once labels are feasible enough to
fit. That is the interior merit minimum appearing — evidence for the mechanism,
not merely the effect. Transition is around label eq_l2 ~ 1.3.

`k_loss` stays pinned at 950 across every tier: **SL loss never says stop, at any
label quality.**

Upper tiers (3.0/4.0/10.0) test the falling half of the inverted U. If the ratio
keeps rising instead, the story is monotone in label feasibility rather than an
inverted U, and the motivation must be worded accordingly.

**SL loss never says stop** — its minimum is the last checkpoint in 7 of 8 pools.
So trusting the supervised objective means training to convergence, which at
maxt2.0 costs **2.4–3.3x task merit**, consistently over 4 seeds.

Consequence for the writeup: **`conv` IS SL-loss selection.** Relabel it as such.
It stops being an arbitrary "why k=950?" baseline and becomes the principled
thing a practitioner does absent a task-faithful signal, which makes
merit-vs-conv the direct test of the thesis.

### Label quality is really degree of infeasibility

| maxt | eq_l2 | ineq_max | success% |
|---|---|---|---|
| 0.5 | 3.79 | 32.2 | 0% |
| 1.0 | 2.92 | 24.4 | 0% |
| 1.5 | 1.33 | 12.1 | 0% |
| 2.0 | 0.403 | 4.80 | 0% |
| 3.0 | 0.0169 | 0.893 | 1.9% |
| 4.0 | 1.5e-14 | 0.270 | 40.6% |
| 10.0 | 1.5e-14 | 1.5e-09 | 100% |

### Amend the motivation: inverted U, not monotone

The naive reading ("worse labels -> more overfitting -> more need for merit") is
**contradicted**: maxt0.5 (eq_l2 3.79) gives only 1.07x while maxt2.0 (eq_l2
0.40) gives 2.71x. Mechanism: overfitting harm requires the model to be *able*
to fit the labels. maxt0.5 labels are too broken to fit, so merit stays flat and
there is nothing to select. maxt10 labels are exact, so fitting them aligns with
the task and conv is fine again. Peak value at **intermediate** inexactness.

This is a stronger claim than the monotone version because it predicts where the
method helps and where it will not. Job 18923016 builds maxt1.0/1.5 pools to
fill in the curve; tiers 3.0/4.0/10.0 pending QOS headroom.

### Open gap in this argument

The loss used is **training** loss on cheap labels, so monotone decrease is close
to guaranteed and the comparison is a bit too easy. A rigorous version holds out
~200 of the 800 labeled instances and selects on *held-out* cheap-label loss.
First thing a reviewer will ask; cheap to add.

---

## 4c. Budget audit — every arm at the same total offline compute

`budget_audit.py <manifest> <ssl_rate>` recomputes
`gen + 0.16*k + ssl_rate*epochs` per arm and flags any deviation from nominal B
above one SSL epoch of rounding.

```
FSNet  (56 arms): max |err| = 1.60s   tol 3.30s   VERDICT: all arms budget-matched
adaPen (28 arms): max |err| = 0.08s   tol 0.18s   VERDICT: all arms budget-matched
```

Vanilla has no gen and no SL by construction (it uses no labels), so its whole
budget goes to SSL. Intended asymmetry, not an error.

### Where merit's budget mechanism can matter

| B | regime | post-gen remainder | conv SL share | merit epoch gain |
|---|---|---|---|---|
| 2000 | q=2.0 | 381.9s | **39.8%** | **+53.9%** |
| 2800 | q=2.0 | 1181.9s | 12.9% | +12.0% |
| 2000 | q=0.5 | 1603.7s | 9.5% | +2.9% |
| 2800 | q=0.5 | 2403.7s | 6.3% | +1.9% |

### SL merit gap does not transfer proportionally to SSL

| regime | SL merit gap (conv vs k*) | observed SSL gap | direction |
|---|---|---|---|
| q=0.5 | 1.07x | 1.25x | **amplified** |
| q=2.0 | 2.71x | 1.15x | **attenuated** |

At q=2.0, SSL partially recovers from a bad init, so a large SL-side advantage
shrinks downstream. At q=0.5 a small SL-side difference grows. So SL merit is
*not* a proportional predictor of downstream benefit — worth stating explicitly,
and a caution against arguing the SSL claim from the SL numbers alone.

### Consistency requirement (per Khai)

If merit selects the same k as conv, their SSL results must be equal. Satisfied
by construction: same k -> same SL cost -> same epoch count -> same run. Under
the robust rule k* never equals 950 in our pools, so no direct case arises. The
*near*-equality case is the real test and currently **fails**: k=900 vs k=950
gave 48.94 vs 30.81 under cosine (1.6x apart for near-identical inits). Job
18918789 measures this sigma under const LR. Until it shrinks, no downstream gap
below ~1.6x is interpretable — including the 1.09–1.31x merit-vs-conv result.

---

## 4d. n=4 RESULT (B=2000, FSNet, const LR) — corrected

**Seed 0 was unrepresentative.** The earlier seed-0-only numbers (merit 3.39x over
vanilla) did not survive 4 seeds.

| comparison | geomean | per-seed | verdict |
|---|---|---|---|
| q=0.5 vs conv | 1.19x | 1.31, 1.20, 1.07, 1.19 | **wins 4/4** |
| q=0.5 vs early | 1.57x | 4.65, 1.06, 1.09, 1.11 | **wins 4/4** |
| q=0.5 vs vanilla | 1.09x | 3.39, 0.51, 0.70, 1.18 | loses 2/4 |
| q=2.0 vs conv | 1.12x | 1.09, 1.21, 1.41, 0.86 | wins 3/4 |
| q=2.0 vs early | 0.78x | 1.12, 0.50, 0.88, 0.75 | **loses 3/4** |
| q=2.0 vs vanilla | 0.86x | 1.11, 0.57, 0.90, 0.97 | **loses 3/4** |

**Surviving claim: merit > conv** (= merit beats SL-loss selection), 4/4 at q=0.5
and 3/4 at q=2.0. That is the direct test of the thesis. **"Merit beats all three"
is refuted at B=2000**: vanilla and early each beat merit on 3 of 4 seeds at q=2.0.

### Why vanilla wins at q=2.0 — gen's budget share

At B=2000, q=2.0, **gen alone is 1618s = 81% of the budget**, so WS arms get
70-111 SSL epochs against vanilla's 606 (5-8x disadvantage). No init advantage
survives that. Monotone in gen share:

| cell | gen share of B | WS vs vanilla epochs | merit vs vanilla |
|---|---|---|---|
| q=0.5, B=2000 | 20% | 450 vs 606 (1.3x) | 1.09x (tie) |
| q=2.0, B=2000 | 81% | 106 vs 606 (5.7x) | 0.86x (loses) |

Same precondition that killed the penalty family, now appearing *inside* FSNet:
warm-starting only competes with vanilla when label generation is a small
fraction of the budget.

---

## 4e. Pure SSL variance (the noise floor) — job 18918789

Identical checkpoint, SSL seed varied only, B=2000:

| arm | identical ckpt? | n | sd/mean | max/min from seed alone |
|---|---|---|---|---|
| vanilla | no — fresh init per seed | 4 | **33.7%** | **2.68x** |
| q=0.5 early (k=100) | yes | 2 | 12.3% | 1.28x |
| q=0.5 merit (k=750) | yes | 2 | 7.0% | 1.15x |

**WS arms are low-variance (7-12%); vanilla is high-variance (34%).** WS arms
inherit fixed weights so only shuffling/dropout vary; vanilla re-rolls the init.
Use per-arm sigma, never pooled.

Consequences:

1. **The merit-vs-vanilla split verdicts are largely vanilla's noise.** Per-seed
   ratios span 6.6x (0.51-3.39) while vanilla alone contributes 2.68x.
   Vanilla comparisons need many more seeds than WS-vs-WS ones.
2. **merit-vs-conv is at the edge**: 1.19x / 1.12x against a ~1.15x WS floor.
   Consistent in sign (4/4, 3/4) but not safely above noise. More seeds needed.

### FINAL sigma (n=4 all arms, job 18918789 complete)

| regime | arm | mean | sd/mean | max/min from seed alone |
|---|---|---|---|---|
| — | vanilla | 72.24 | 28.9% | 2.39x |
| 0.5 | merit | **27.18** | 24.3% | 2.04x |
| 0.5 | conv | 32.53 | 6.4% | 1.18x |
| 0.5 | early | 103.00 | 11.4% | 1.33x |
| 2.0 | merit | **76.77** | 2.5% | 1.06x |
| 2.0 | conv | 87.10 | 2.1% | 1.06x |
| 2.0 | early | 62.84 | **59.9%** | **5.93x** |

Variance is **heterogeneous per arm** (2% to 60%), so there is NO single noise
floor — per-comparison significance is the only honest read. Never pool sigma.

This design fixes the SL model and varies only the SSL seed, so it is a properly
controlled comparison:

| regime | comparison | diff ± SE | t | verdict |
|---|---|---|---|---|
| 0.5 | merit vs vanilla | 45.1 ± 10.9 | **4.1** | merit better |
| 0.5 | merit vs early | 75.8 ± 6.8 | **11.2** | merit better |
| 0.5 | merit vs conv | 5.35 ± 3.46 | 1.55 | merit better, n.s. |
| 2.0 | merit vs conv | 10.33 ± 1.34 | **7.7** | merit better |
| 2.0 | merit vs early | -13.9 ± 18.8 | -0.74 | tie (early sigma 60%) |
| 2.0 | merit vs vanilla | -4.5 ± 10.5 | -0.43 | tie |

**Merit is never significantly worse than any arm.** At q=0.5 it beats vanilla and
early decisively, conv directionally; at q=2.0 it beats conv decisively and ties
the others.

Why this disagrees with the paired sweep (§4d): the paired design let k* vary per
seed AND was hostage to `early`'s 5.93x swing. With sigma that large, win/loss
counts flip on nothing. The fixed-SL design is the cleaner comparison but
excludes k*-selection variability by construction, so it understates real-world
variance. Report both, and say which is which.

### The two competing effects (this is the synthesis)

| | q=0.5 | q=2.0 |
|---|---|---|
| SL-side selection value | 1.07x (low) | 2.71x (high) |
| gen share of budget | 20% (WS well fed) | 81% (WS starved) |
| net merit vs vanilla | **wins, t=4.1** | tie |
| net merit vs conv | +5.4, t=1.55 | **wins, t=7.7** |

Selection matters more with better labels, but better labels cost more gen and
starve SSL. The sweet spot needs labels cheap enough not to starve WS yet good
enough that selection matters. **maxt1.5** (gen 1171s, selection value 1.74x) is
the untested middle and is the obvious next cell to run.

Caveat: all of the above is const LR, so each arm's final-checkpoint readout is
~1.42x off its own best (see 4f). The cosine re-run (18926780) supersedes it.

---

## 4g. maxt1.5 cell — RUN, and it went against the hypothesis

Reporting decision (Khai): **primary FSNet table = maxt0.5 + maxt2.0** (the original
two-regime design). maxt1.5 was an exploratory cell added mid-session. It was run
to completion and is recorded here; disclose it in the writeup rather than delete
it, since the exclusion decision followed the result.

FSNet | cosine | B=2800 | maxt1.5 (gen 1171.5s = 42% of B) | test merit, lower better

| seed | merit(k*) | early(100) | conv(950) | vanilla | vs van | vs early | vs conv |
|---|---|---|---|---|---|---|---|
| 0 | 105.79 | 112.24 | **73.28** | 91.4 | 0.86 | 1.06 | **0.69** |
| 1 | 80.05 | **29.40** | 52.28 | 76.4 | 0.96 | 0.37 | **0.65** |
| 2 | 94.76 | 105.63 | **60.06** | 94.4 | 1.00 | 1.11 | **0.63** |
| 3 | 65.22 | 97.03 | (cancelled) | 88.8 | 1.36 | 1.49 | — |

```
merit vs conv     n=3  geomean 0.66x  [merit LOSES ALL]  0.69, 0.65, 0.63
merit vs vanilla  n=4  geomean 1.03x  [split 1/4]        0.86, 0.96, 1.00, 1.36
merit vs early    n=4  geomean 0.90x  [split 3/4]        1.06, 0.37, 1.11, 1.49
```

merit loses to conv on all 3 complete seeds, and the spread (0.63-0.69) is the
**tightest** of any comparison measured this session — signal, not noise. conv is
~1.5x better as a warm start. This cell was predicted to be the MOST favorable
(selection value 1.74x, gen only 42% of B), so it is a genuine test that failed.

### merit vs conv across all three regimes (cosine)

| regime | geomean | verdict |
|---|---|---|
| maxt0.5 | 1.15x | split 2/3 |
| maxt1.5 | **0.66x** | **loses 3/3** |
| maxt2.0 | 0.99x | split 2/3 |

**Dropping maxt1.5 does not rescue the conv claim** — it is unresolved at 0.5 and
2.0 as well. What the two kept regimes DO support: at maxt2.0, merit beats vanilla
3/3 (2.32x) and early 3/3 (1.47x).

### Interpretation: the SL merit advantage ANTI-transfers

SL-side says merit selection is worth 1.74x at maxt1.5; downstream it is 0.66x.
So the checkpoint that is a better standalone solver by task merit is a *worse*
initialization. Plausible mechanism: warm-starting wants a well-developed feature
extractor, not the best standalone solver. merit stops SL at k=150-400, leaving a
less-developed network that SSL must build up anyway; conv at k=950 has richer
features despite worse standalone merit.

**Consequence for the thesis.** Split the claim:
- merit IS the right criterion for selecting a **deployable SL model** (7 tiers,
  4 seeds, up to 2.95x over SL-loss selection). Solid.
- merit is NOT established as the right criterion for selecting a **warm start**;
  transfer is unreliable and at maxt1.5 inverted.

A warm-start rule that wins would need a signal measuring *transferability* rather
than standalone quality. On current evidence "always conv" would beat merit as a
warm-start rule.

---

## 5. Methodology gotchas (all cost real time this session)

1. **`eta_min` was 1e-3, larger than every configured LR.** `CosineAnnealingLR`
   with `eta_min > base_lr` *rises* toward eta_min instead of decaying, so every
   pre-2026-07-26 run trained with an increasing LR. Fixed in the working tree
   (`utils/trainer.py:708`, default 1e-6, `--eta_min` / `--lr_schedule` exposed
   in `main.py:73-76`). **All pre-07-26 checkpoints are suspect.**
2. **Cosine `T_max = num_epochs` is a confound.** Arms differ in epoch count, so
   under cosine they also differ in LR trajectory (penalty spanned 1277–11111,
   an 8.7x spread). Use `--constant_lr` for budget sweeps so epochs are the only
   axis.
3. **Saves fail silently.** `utils/trainer.py:916-970` wraps every save in
   `try/except`, prints `x Error saving ...`, and returns 0 — hence the
   `Files saved (or attempted):` banner. Combined with sbatch not checking
   python's exit code, a run that loses all output reports `COMPLETED 0:0`.
   This destroyed 1.8 GPU-h of const-LR runs and all 6 adaptive_penalty arms.
   Scripts now propagate exit codes and grep for `Error saving` /
   `Detailed results saved`. **Consider making `trainer.py` re-raise.**
4. **Dir names do not identify the SL seed.** The `_finetune_<ts>_sup_pen_model_<k>`
   tag records only the SL run's *timestamp*, and all four q=0.5 pools share
   `20260726-125645` (one array launch). Paired-seed and fixed-SL runs are
   therefore indistinguishable by path. Match on `config['checkpoint']` inside
   `results.pkl` instead — that is authoritative.
5. **Two filesystems, don't confuse them.** `results/` is a symlink to
   `/orcd/pool/007/khain/FSNet/results` (1.0T pool). The repo is on
   `10.1.223.4:/home` (182G/195G/200G user quota). The `Errno 122` failures came
   from the *pool* filling, not the home quota. Pool went 342G -> 38G after
   pruning; 304G free now.
6. **`seed` does not affect the data split.** `utils/optimization_utils.py:30-37`
   uses fixed indices: train `[:train_size]`, val `[7000:8000]`, test
   `[8000:10000]`; the `seed` arg is accepted but unused there. Seed controls
   weight init, shuffling, dropout (`main.py:199-201`). Implications: sharing one
   SL model across SSL seeds is leak-free; SL's 800 labeled instances are a clean
   subset of SSL's 7000; and for warm-start arms the SSL seed only changes
   shuffling/dropout, so most between-seed spread comes from the *different SL
   model*, not from SSL.
7. **Heteroscedastic error bars.** Vanilla gets a fresh random init per seed;
   WS arms inherit fixed weights. Vanilla will have systematically larger sigma,
   so use per-arm sigma rather than a pooled estimate.
8. **`--save_intermediate` is `type=bool`** (`main.py:33`), so *any* non-empty
   value is truthy — `--save_intermediate False` also enables it.
9. **`lr_decay` / `lr_decay_step`** in the penalty and sup_partial configs are
   dead unless `--lr_schedule step` is passed.

---

## 6. Experiment inventory

| file | purpose |
|---|---|
| `run_sl_pretrain_800.sh` | 8 SL pools (2 regimes x 4 seeds), 1000 ep, ckpt every 50 |
| `gen_budget_manifest.py` | emits `budget_manifest.tsv` (56 rows: B=2000 idx 0-27, B=2800 idx 28-55) |
| `budget_manifest_fixedSL.tsv` | single SL model (seed-0 pools), SSL seeds 1-3 — variance diagnostic |
| `budget_manifest_adapen.tsv` | adaptive_penalty, B=2000, 4 seeds (0.18 s/ep) |
| `run_fsnet_budget_sweep.sh` | array driver; `METHOD` / `MANIFEST` / `SCHED` env overrides |
| `analyze_budget_sweep.py` | matching + mean±sigma + paired per-seed ratios + CSV export |
| `sl_selection_value.py` | merit-vs-SL-loss selection value per tier (SL-only, no SSL) |
| `budget_audit.py` | verifies every arm charged the same total offline budget |
| `analysis/fsnet_cosine.csv` | 36 runs — REPORTED config (cosine, B=2800) |
| `analysis/fsnet_constlr.csv` | 57 runs — superseded const-LR sweeps + sigma study |
| `analysis/fsnet_paired.csv` | 34 runs — earlier const-LR paired snapshot |

SL pools (fresh, post-`eta_min`-fix, 20 ckpts each):
`20260726-125645_MLP_sup_pen_seed{0..3}_..._subopt_3_0.5_...` and
`20260726-125720_MLP_sup_pen_seed{0..3}_..._subopt_3_2.0_...`

Refresh tables:
```bash
python analyze_budget_sweep.py budget_manifest.tsv --csv=analysis/fsnet_paired.csv
python analyze_budget_sweep.py budget_manifest_fixedSL.tsv --csv=analysis/fsnet_fixedSL.csv
SWEEP_METHOD=adaptive_penalty python analyze_budget_sweep.py \
    budget_manifest_adapen.tsv --csv=analysis/adapen.csv
```

### Jobs in flight (as of writing)

| job | what | state |
|---|---|---|
| 18918574 | FSNet B=2000 paired, seeds 0-3 | mostly done |
| 18918628 | FSNet B=2800 paired, seeds 0-3 | running |
| 18918789 | FSNet fixed-SL, SSL seeds 1-3 | running |
| 18921377 | adaptive_penalty B=2000, seed-0 screen | pending |
| 18923016 | SL pools maxt1.0/1.5, 4 seeds (inverted-U curve) | running |
| 18924184 | SL pools maxt3.0/4.0/10.0, 4 seeds | queued |
| ~~18921377~~ | adaptive_penalty B=2000 | **cancelled** — arms would be saturated |

Note: full 28-task arrays hit `QOSMaxSubmitJobPerUserLimit` (partition QOS cap,
not the 500-job account limit). Submit in smaller chunks as arrays drain.

---

## 7. Open questions

1. **Does merit beat `early`?** The one comparison currently failing. Driven by
   q=2.0 seed 1, where early scored 2x better than merit. Needs seeds 2-3.
2. **What is pure SSL sigma?** Job 18918789 answers it. If sigma is ~1.3x then
   even the merit-vs-conv result is inside noise and more seeds are required.
   Everything above is contingent on this.
3. **Is the q=0.5 reversal from const LR or from robust k*?** Disentangle with
   argmin k*=900 under const LR.
4. **Budget sweet spot.** B=2000 looks over-tight at q=2.0; B=2800 separated
   better. Consider a sensitivity curve over B rather than a single point.
5. **Claim shape.** "Merit strictly dominates in both regimes" is probably not
   reachable — at q=0.5 both of merit's mechanisms are structurally absent under
   a shared budget. Two honest alternatives:
   - **per-regime budget** (at N=2000, q=0.5, B=1500: conv 40 ep vs merit 94 ep,
     a **2.35x** win), which costs the cross-regime transfer framing; or
   - **report a sensitivity curve** of merit's advantage vs. SL's share of the
     offline budget, stating the threshold above which merit wins. Converts
     "does merit win?" into "when does merit win?" — answerable, and not
     vulnerable to a cherry-picked-operating-point critique.
6. **penalty family viability** — pending 18921377.

---

# Session 2 (2026-07-26, later): penalty family under a matched budget

Goal set by Khai: make **merit beat vanilla for the penalty method at equal total
offline budget**, with labels fixed at **800 instances @ maxt0.5** (gen 396.3 s).

All runs below charge the reference rates regardless of GPU: SL 0.16 s/ep,
penalty SSL 0.18 s/ep, adaptive_penalty SSL 0.18 s/ep, gen 396.3 s.
**Merit convention here is `obj + 1e5*(eq_l1 + ineq_l1)`.** The `merit_mean` field
in the CSVs uses a 1e6 weight, exactly 10x these values; ratios are unaffected.

## Result: merit does not beat vanilla, across four independent designs

| design | method | B (s) | n | van/merit | merit wins |
|---|---|---:|---:|---:|---|
| per-seed SL pools | penalty | 1100 | 8 | 0.9469x | 3/8 |
| fixed SL (seed-0 pool, k*=750) | penalty | 1100 | 8 | 0.9631x | 2/8 |
| per-seed SL pools | adaptive_penalty | 952 | 4 | 0.884x | 0/4 |
| equal-epoch control (merit3000) | adaptive_penalty | n/a | 4 | 0.938x | 0/4 |

The equal-epoch control matters: giving the merit arm the same 3000 epochs as the
`early` arm (deliberately over budget) still loses 4/4. The loss is therefore in
the quality of the SL init, not in budget accounting.

CSVs: `penalty_maxt05_B1100_n8.csv` (32 runs), `penalty_maxt05_fixedSL_n8.csv`
(32), `adapen_meritswap_maxt05_B952.csv` (20), `penalty_maxt05_B1100.csv` (16).

## The one effect that replicated: earlier checkpoints are better inits

Per-seed SL, n=8, paired: **early beats conv 8/8 (1.061x)** and **early beats
merit 7/8 (1.059x)**. Ordering is monotone in k: k=100 > k*=500-850 > k=950. The
more SL you invest, the worse the SSL init. This is the opposite of the thesis and
it is the most robust downstream finding in the study.

## Mechanism: the cosine restart erases the warm start

Validation `stop_merit` along training (n=8 per arm, penalty, B=1100):

- vanilla: 74,640 @ ep1550 -> 69,879 @ ep3050 -> flat (70,713 final). Saturates ~3000.
- merit:   73,546 @ ep800 -> **rises to 76,756 @ ep1600** -> 71,797 final.

Merit's curve going *up* for 800 epochs is the failure. Every arm starts a fresh
full-amplitude warmup+cosine, so the SL init is hit with lr back at 1e-4 and is
largely destroyed, then partially rebuilt using the rest of the budget. It never
returns to vanilla's level. This also explains the k-ordering above: the least
invested checkpoint (k=100) has the least to lose.

**Open test:** symmetric LR sweep, `lr in {1e-4, 3e-5, 1e-5}` applied to *both*
arms, 4 seeds, job 18947417, manifest `budget_manifest_pen05_lrsweep.tsv`.
If the diagnosis holds, merit at low LR should not show the hump. NOT YET RUN TO
COMPLETION as of this writing.

## Why no choice of B can rescue it (given 800 labels @ maxt0.5)

Warm-start overhead is `396.3 + 0.16*k*` = ~510 s at k*=712, i.e. ~2830 SSL epochs
handed to vanilla for free. Vanilla saturates by ~3000 epochs (540 s). So:

- `B < ~1050`: merit is epoch-starved *and* still behind.
- `B >= ~1050`: both arms are past saturation, and merit's asymptote (71,797) is
  worse than vanilla's (70,713).

There is no B in between. Merit can only win by reaching a *better* asymptote,
which under the current schedule it does not. Reducing gen cost (fewer labels)
would change this arithmetic, but Khai fixed labels at 800 @ maxt0.5.

## Label-free SL-pool selection does not predict downstream quality

Ranking the eight maxt0.5 pools by validation `stop_merit` at their own k*:

| rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| seed | 3 | 1 | 7 | 5 | 6 | 2 | 4 | **0** |

An honest label-free rule picks **seed 3**, whose merit arm loses to vanilla
(0.9411x). Seed **0**, ranked last, is the only seed where merit beats vanilla,
early and conv simultaneously. The correlation is inverted in this sample, so
"select the best SL pretrain" does not transfer here.

Confirmed directly: the fixed-SL design took seed-0's pool (the selection winner)
and ran 8 fresh SSL seeds. Merit went 2/8, geomean 0.9631x. Seed 0's sweep was
SSL-seed luck, not pool quality.

## Selected-seed table

`analysis/FINAL_TABLE_penalty_maxt05.md` holds both the all-8 table and a
seeds-{0,1,2} table requested by Khai. The selected table shows merit ahead by
**704** (7.058e4 vs 7.128e4) against a **+-3.3e3** spread, i.e. the standard
deviation is 4.7x the reported effect. The file states in-table that those three
seeds were chosen because merit wins on them; that line must travel with the
numbers.

## Note on the earlier adaptive_penalty "win"

The recorded AdaPen win (ckpt 100, 3000 ep, lr 5e-4) used SL pool
`20260116-022657`, which predates the eta_min fix and was trained with the buggy
rising schedule; only ckpts 50-400 were kept there, and merit on that pool selects
k=660, so the merit arm was never testable on it. Its baseline was also a "Van
continue" resuming from a paper SSL checkpoint rather than a cold start. Rerun on
the corrected July pool against a cold vanilla at matched budget, every warm-start
arm loses 4/4.

Also: with `eqW=50` given to vanilla too (it had previously been applied to the
warm-start arm only), the penalty screen's apparent win disappears.

---

# Session 3 (2026-07-27): AdaPen closed out; the penalty-family break-even

## AdaPen replication — negative, n=8

Seeds 4-7 added to the five cells that had looked like wins at n=4
(`mf_rep_eq10/50/2000.tsv`, jobs 18970461/62/63). Nothing crosses 1.0.

| eqW/ineqW | lr | n=4 geomean | n=8 geomean | wins |
|---:|---:|---:|---:|---:|
| 50/10 | 3e-5 | 0.9860 | **0.9896** | 3/8 |
| 10/10 | 3e-5 | 0.9752 | 0.9828 | 2/8 |
| 10/10 | 1e-4 | 0.9627 | 0.9793 | 3/8 |
| 2000/500 | 1e-5 | 0.9741 | 0.9702 | 1/8 |
| 10/10 | 5e-4 | 0.9442 | 0.9001 | 1/8 |

Across LR 5e-4 -> 3e-6, eqW 10 -> 2000, checkpoint E 50 -> 950 and two SL designs,
**no AdaPen configuration has a warm-start arm beating vanilla at n >= 4.** The
n=4 "wins" were single seeds inside a losing geomean.

The E=200 row that appeared in the draft budget table (4.680e4 vs vanilla 4.698e4)
is n=4, 2/4 seeds, 0.4% against vanilla's own 1.4% sigma, **and it is not the
checkpoint the merit rule selects** (k* picks 500-800; at k*=500 the arm loses).
It was never replicated at seeds 4-7.

## The break-even, which decides the whole penalty family

Warm-start value measured directly: take the WS validation `stop_merit` curve and
ask which vanilla epoch reaches the same value.

| WS arm (lr 1e-4, seed 0) | max epochs saved vs vanilla | seconds saved |
|---|---:|---:|
| k=750 | ~150 | **27 s** |
| k=100 | ~120 (excluding one noise dip) | ~22 s |

**A warm start is worth ~20-30 s of penalty SSL.** Label generation for 800
instances @ maxt0.5 costs **396.3 s**. The warm start must therefore be ~15x
cheaper than it is before a matched-budget win is arithmetically possible;
break-even is `gen + SL <~ 27 s`, i.e. **~50 labels @ maxt0.5**.

This is why no B works, and it is independent of B by construction.

## Better labels do not fix it: the SL init is 5.6x worse than vanilla's asymptote

Best validation `stop_merit` over each SL checkpoint pool (800 labels, same val
set and same merit formula as the SSL runs):

| maxt | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 10.0 |
|---|---|---|---|---|---|---|---|
| best over pool | **2.24e5** | 2.32e5 | 2.99e5 | 3.49e5 | 2.98e5 | 2.70e5 | 2.76e5 |
| at k=950 | 2.43e5 | 2.54e5 | 5.32e5 | 9.48e5 | 8.78e5 | 6.06e5 | 5.59e5 |

Vanilla penalty reaches **3.97e4** (lr 1e-5, 6100 ep) / 6.22e4 (lr 1e-4). So the
best SL init at *any* label quality is 5.6x worse than what vanilla penalty gets
to on its own, and label quality does not help — maxt2.0 is the *worst* init
(overfits 800 instances). Warm-starting can only ever save the early descent.

## LR sweep: refutes the "cosine restart erases the warm start" diagnosis

Both arms at the same LR, penalty, B=1100, maxt0.5:

| lr | n | van/merit | wins |
|---:|---:|---:|---:|
| 1e-4 | 8 | 0.9773 | 4/8 |
| 3e-5 | 4 | 0.8948 | 0/4 |
| 1e-5 | 2 | 0.9397 | 0/2 |

Lower LR makes the warm start **worse**, not better. It also makes vanilla much
better in absolute terms (3.98e5 at 1e-5 vs 6.90e5 at 1e-4, merit_mean scale), so
the paper's vanilla-penalty baseline should be lr=1e-5, not 1e-4 — which widens
the gap the warm start has to close.

## Vanilla-vs-arm pairing at B=1100 (n=8, both designs)

Previously only van/merit was tabulated. Full pairing:

| design | conv | early (k=100) | merit (k*) |
|---|---:|---:|---:|
| per-seed SL | 0.9449 (3/8) | **1.0028 (3/8)** | 0.9468 (3/8) |
| fixed SL | 0.9487 (2/8) | 0.9930 (2/8) | 0.9630 (2/8) |

`early` is a tie with vanilla, not a win, and it is not what the merit rule picks.

## Tooling

`analysis/adapen_rep_summary.py` — pairs vanilla vs warm-start by parsing lr /
eq / ineq / checkpoint out of the run directory name. Needed because
`analyze_budget_sweep.py` keys on `(seed, epochs, checkpoint)`, which collides
when the same manifest row is run at several LRs or penalty weights.

```bash
python analysis/adapen_rep_summary.py --method penalty --since 20260726-000000
```

## Exhaustive checkpoint-resolved scan for any budget-matched WS win

`analysis/build_run_cache.py` caches all 293 post-fix penalty-family runs to
`analysis/run_cache.csv`; `analysis/find_ws_wins.py` pairs every warm-start run
against a vanilla run sharing method / lr / eq / ineq / schedule / seed / SL pool
whose implied budget agrees to within one SSL epoch. **Each checkpoint k is its own
arm** — collapsing all warm starts into one "merit" arm hid the E-sweep cells.

Every cell with geomean > 1 (ratio = vanilla / warm-start, so > 1 means WS wins):

| method | lr | eqW | B | k | n | van/ws | wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| penalty | 1e-4 | 50 | 1100 | 500 | 2 | 1.0141 | 2/2 — **both runs are seed 2** |
| penalty | 1e-4 | 50 | 1100 | 800 | 2 | 1.0137 | 2/2 — **both runs are seed 1** |
| adaptive_penalty | 3e-5 | 50 | 952 | **100** | 4 | **1.0100** | 2/4 |
| penalty | 1e-4 | 50 | 1100 | 100 | 12 | 1.0059 | 4/12 (two designs pooled) |
| adaptive_penalty | 3e-5 | 50 | 952 | **200** | 4 | **1.0039** | 2/4 |

Nothing else clears 1.0 at any n. The two penalty cells at 1.014 are single seeds
(k=500 and k=800 are those seeds' own k*, already inside the merit aggregate).

**So the only surviving candidates are AdaPen E=100 and E=200 at lr 3e-5, eqW 50,
B=952 — and neither has ever been run at seeds 4-7.** The n=8 replication covered
the *merit-k\** cells, not these. 16 runs, ~20 min, would settle the drafted table row.

### Gaps found by the same scan

- **penalty at B=480 (the row in the draft table) has n=1 and loses.** WS k=100,
  seed 0, 355 ep -> merit 9.81e5, against a budget-matched vanilla (2667 ep,
  B=480.1) at 8.12e5, i.e. **0.83x**. Vanilla exists at seeds 0-3; the WS arm was
  never run at seeds 1-3.
- **penalty at B=778** has three WS checkpoints (k=100/200/950, seed 0) and **no
  budget-matched vanilla at all** (would need 4322 epochs).
- 238 further run dirs (2026-01/03/05, all of 07-25, and the const-LR sets) are
  excluded: they predate the `eta_min` fix and trained with a rising LR. The
  historical AdaPen "win" lives in that set and was already discredited above.
