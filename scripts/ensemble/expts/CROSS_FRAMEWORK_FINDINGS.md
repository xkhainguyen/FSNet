# Cross-framework verification: when does multi-start / "free ensembling" actually help in L2O?

**Question.** On amortized constrained optimization, pre-repair multi-start ("free
ensembling" — perturb the NN output K times and run repair on each, then pick the
best by merit) was originally reported as a near-zero-cost ensemble alternative,
giving ~60% merit reduction at K=100 on FSNet over single-shot inference. Is this
gain *real* (a genuine functional-diversity benefit) or a *repair-operator
under-convergence artifact* (the multi-start papers over a badly-tuned solver)?

**Verdict (6 testbeds tested).** The big perturbation gains are **convergence
artifacts on one of two axes**; every genuine convergence-surviving residual is
small (~1%), *including the multimodal case once trained to convergence*.

1. **Under-converged iterative repair** (FSNet/L-BFGS, DC3/grad-SGD, Πnet/ADMM,
   convex objectives) → **artifact**. Collapses to noise once the repair
   converges; fixing the repair budget is 10–20× cheaper. *Most published "free
   ensemble" results on convex benchmarks live here.*
2. **Undertrained network** (HardNet at 200 epochs looked like a 2× win) →
   **artifact**. Training to convergence (1500 ep) collapses the gain
   101% → 1.1%. Same mechanism as (1) but on the *network* axis: perturbation
   papers over a not-yet-converged NN. Train longer instead.
3. **Direction-dependent repair** (Bisection-Projection radial bisection ≈0.4%;
   HardNet's non-Euclidean pinv/ReLU correction ≈0.5%) → **real but small**;
   survives convergence because the repaired point depends on approach direction.
4. **Nonconvex objective** (HardNet `pᵀsin(y)`, ≈0.6% increment at convergence)
   → **real but tiny** — not the large effect the undertrained run suggested.

5. **Multimodal / disconnected feasible set** (§6) → **real but small (~1%)**.
   The *most* genuine of the convergence-surviving effects: a continuous net
   cannot perfectly represent the discontinuous argmin-component map, so a ~5.6%
   routing-error floor remains even at convergence (it flattens 240k→480k). But
   the headline "94%" was undertraining + 2D random-search — proper LR-decay
   training more than halved the apparent floor (13%→5.6%) and shrank the K=100
   gain 0.042→0.011. The surviving gain is ≈1% of the objective. The
   connected-vs-disconnected *distinction* is real (a convex single ball needs
   no ensembling — Control B), but it does **not** translate into a large
   convergence-surviving win.

A convex objective with a converged Euclidean projection **and a converged
network** gets ~zero genuine benefit. Every effect that survives full
convergence — non-Euclidean repair (3), nonconvex objective (4), multimodal
routing (5) — is order ~1%. **No testbed shows a large convergence-surviving
perturbation/ensemble gain.** The headline numbers (60%, 2×, 94%) were all
under-convergence on the repair or the network.

> **Correction note.** An earlier draft of this doc claimed HardNet showed a
> "real, large nonconvex-objective win (2×)." The 1500-epoch controls (§5)
> showed that was ~95% undertraining. Logged so the error isn't silently
> overwritten — it's the same class of mistake (mistaking under-convergence for
> a genuine ensembling effect) that the whole study is about, just on the
> network instead of the repair.

Frameworks: FSNet (this repo, L-BFGS), DC3 (this repo, grad-SGD), Πnet (Terpin
et al. 2025, ADMM), Bisection-Projection (Liang et al. 2025, radial bisection),
HardNet (Min & Azizan, NeurIPS 2024, closed-form affine projection).

---

## 1. FSNet (this repo) — L-BFGS repair

The FSNet repair (`utils/lbfgs.py:nondiff_lbfgs_solve`) minimises
`||eq_resid||² + ||ineq_resid||²` via L-BFGS. The original convergence check was
"val OR grad", scalar-reduced across the batch — so the loop could early-exit
as soon as *any* per-sample val or grad satisfied tolerance, leaving other
samples under-converged. We added a `per_sample_lbfgs` flag (default False)
that enables (a) per-sample line search and (b) val-AND-grad per-sample
convergence.

| Config (nonsmooth SOCP, seed 0)             | per_sample=0 | per_sample=1 |
|---------------------------------------------|--------------|--------------|
| Single, K=1 baseline                        | Merit 84.7   | Merit 16.5   |
| Single, K=100 ε=0.10                        | Merit 30.5   | Merit 16.5   |
| Perturbation gain (relative)                | **−64%**     | **0%**       |
| VE M=5 ensemble, best_merit, K=1            | Merit 29.7   | Merit 16.4   |
| VE M=5 ensemble, K=100 ε=0.10               | Merit 14.6   | Merit 16.4   |

Same checkpoint, different convergence-check semantics. With the corrected
per-sample convergence, perturbation gives zero benefit. The previously-reported
gain was the L-BFGS scalar early-exit leaving samples un-converged.

## 2. DC3 (this repo) — `grad_steps` repair

The DC3 repair (`utils/optimization_utils.py:grad_steps`) runs a fixed
`max_corr_steps=30` iterations of momentum-SGD with fixed `corr_lr=1e-6` on
the inequality penalty. There is **no convergence check at all** — the loop
always runs exactly 30 iterations regardless of feasibility.

**Sweep A — budget:** No perturbation, sweep `max_corr_steps` from 30 to 3000.

| max_corr_steps | IneqL1 | Merit |
|----------------|--------|-------|
| 30 (default)   | 5.7e-4 | 590.6 |
| 100            | 5.3e-4 | 556.4 |
| 300            | 4.5e-4 | 470.6 |
| 1000           | 2.5e-4 | 270.9 |
| 3000           | 5.2e-5 | 76.2  |

Just letting `grad_steps` run longer gives a 7.7× merit reduction with no
perturbation, no architectural change, no retraining.

**Sweep B — learning rate:** No perturbation, default budget=30, vary `corr_lr`.

| corr_lr | IneqL1 | Merit |
|---------|--------|-------|
| 1e-6 (default) | 5.7e-4 | 590.6 |
| 1e-5    | 4.5e-4 | 472.7 |
| 1e-4    | 4.1e-5 | 65.4  |
| 1e-3    | **0.0** | **24.4** ← exact feasibility |

Just bumping `corr_lr` from 1e-6 to 1e-3 gives a 24× merit reduction and
exact feasibility. Compare: the reported K=100 ε=0.01 perturbation result on
DC3+SOCP was Merit 24.25 — **essentially the same number reached by fixing
the lr alone**. The perturbation was papering over a 1000× under-sized step.

## 3. Πnet (vendored at `third-party/pinet`) — ADMM projection repair

The Πnet repair is Douglas–Rachford ADMM projecting onto
`{y : Ay = b, lb ≤ Cy ≤ ub}`. Defaults are `n_iter_test=100` (max iter) with
`tol=1e-3` (early-exit on max constraint violation < tol).

Trained one HCNN on the DC3 nonconvex QP100 benchmark (50 epochs,
features=[200,200], seed 42) using their `benchmark_config_default.yaml`.
At test time, computed the trunk output (pre-projection), replicated K times,
added ε·N(0,I) noise, ran ADMM projection at varying `n_iter_test`, picked
best by Merit = obj + 1e6·(EqMax + IneqMax).

| n_iter_test | K=1 baseline | best K-perturb | reduction |
|-------------|--------------|----------------|-----------|
| 10          | 7925.6       | -9.32 (K=100 ε=0.10)  | ~100%     |
| 50          | 1629.2       | 369.1 (K=100 ε=0.10)  | 77%       |
| **100 (default)** | **116.5** | **44.6 (K=100 ε=0.10)** | **62%** |
| 500 (5×)    | -11.5563     | -11.5587 (K=20 ε=0.01) | 0.02%    |
| 1000        | -11.5566     | -11.5590              | 0.02%     |
| 3000        | -11.5566     | -11.5590              | 0.02%     |

Same shape. At default budget the perturbation gain is huge (62%) and is almost
entirely closing IneqMax (from 1.28e-4 → 5.6e-5); at 5× budget IneqMax drops
to 1e-10 (the projection has effectively converged), and the perturbation gain
shrinks to within numerical noise.

**Head-to-head: just-bump-budget vs perturb-at-default (Πnet, DC3 nonconvex QP100, seed 42):**

| Strategy                                                | Merit       |
|---------------------------------------------------------|-------------|
| Default budget (n_iter_test=100), K=1 baseline          | 116.52      |
| Default budget, K=100 ε=0.10 perturb (best in grid)     | 44.64       |
| **n_iter_test=500, K=1 baseline (no perturb)**          | **-11.56**  |
| n_iter_test=1000, K=1 baseline                          | -11.56      |

Just increasing the ADMM budget 5× (no perturbation, no architectural change)
beats the best K=100 perturbation result by >50 absolute merit units. The same
pattern as DC3: bumping the repair budget alone dominates multi-start.

---

## 4. Bisection-Projection (vendored at `third-party/Bisection-Projection`) — radial bisection

BP (Liang et al. 2025) uses a structurally DIFFERENT repair operator:
**radial bisection from a predicted feasible interior point to an infeasible
candidate**. Unlike Euclidean projection (L-BFGS, ADMM) or gradient correction
(DC3), bisection follows a 1D line from IP to candidate and finds the boundary
intersection on that line. Bisection converges geometrically (rate 0.9 → 30
iters gives 1e-30 alpha gap), so there's no under-convergence failure mode by
construction.

Pretrained QP100-50-50 model from BP repo. IP per sample computed via cvxpy
feasibility solve (not Y_star, to avoid trivial collapse). Bisection in PARTIAL
space (so equality `Ay=x` is preserved by `complete_partial` at every step).
Metrics on the 70/1024 infeasible samples (where bisection actually runs):

| K | ε    | Obj_inf  | IneqL1_inf | Merit_inf |
|---|------|----------|------------|-----------|
| 1 | 0.00 | -0.2834  | 7.0e-6     | **6.74**  |
| 20  | 0.01 | -0.2838 | 0.0 | -0.2838 |
| 100 | 0.01 | -0.2839 | 0.0 | -0.2839 |
| 100 | 0.05 | **-0.2844** | **0.0** | **-0.2844** |
| 100 | 0.10 | -0.2841 | 0.0 | -0.2841 |
| 100 | 1.00 | -0.2607 | 0.0 | -0.2607 (eps too large) |

**Bisection budget sweep (K=100, ε=0.10):**

| n_steps | Obj_inf | IneqL1_inf | Merit_inf |
|---------|---------|------------|-----------|
| 3       | -0.2840 | 0.0        | -0.2840   |
| 30      | -0.2841 | 0.0        | -0.2841   |
| 300     | -0.2841 | 3e-6       | +2.71 (overshoots) |

Two distinctive properties on BP that the other three frameworks lack:

1. **Perturbation gives a real merit gain (6.74 → -0.28)**, and it's NOT a
   budget artifact. n_steps=3 already converges. Bumping n_steps further
   doesn't tighten the result (and actually hurts at n_steps≥100 because
   alpha approaches 1 and the perturbed candidate's tiny eq residual surfaces).
2. **The "gain" is mostly feasibility-tightening (IneqL1 7e-6 → 0), not
   objective improvement.** K=1 gives Obj=-0.2834, best perturb gives
   Obj=-0.2844 — a 0.0010 improvement. The dramatic merit drop (6.74 → -0.28)
   is because IneqL1=7e-6 hits with weight 1e6.

So BP is the one framework where multi-start exploits the repair *geometry*
(radial vs Euclidean). Different perturbed candidates define different rays
from IP, which hit different boundary points. The gain is **moderate**
(objective improvement ~0.4%) but **real** (survives full bisection
convergence).

## 5. HardNet (vendored at `third-party/hardnet`) — closed-form affine projection

HardNet-Aff (Min & Azizan, NeurIPS 2024) appends a **closed-form, one-shot**
correction: `proj(f,x) = f + pinv(A)·(ReLU(bl−Af) − ReLU(Af−bu))`. There is no
iterative solver — no max_iter, no tolerance, no convergence loop. So the
"under-convergence artifact" explanation is structurally impossible here.

> **Caveat on the operator.** This is NOT the Euclidean projection onto
> `{bl ≤ Af ≤ bu}` (that would be a QP — which is what the sibling file
> `hardnet_cvx.py` does via a `CvxpyLayer`). HardNet-Aff is a single pinv/ReLU
> step that is feasible-by-construction under HardNet-Aff's assumptions, but is
> **direction-dependent**: different `f` map to different feasible points (same
> property as BP's radial bisection). So this framework is NOT a clean
> "Euclidean + converged" control.

HardNet's `opt` benchmark is the DC3 **nonconvex-objective** QP100:
`min ½yᵀQy + pᵀsin(y) s.t. Ay≤b, Cy=x` — the only nonconvex *objective* tested
(FSNet/DC3/Πnet/BP all had convex objectives).

Trained 200 epochs (50 warmup, no projection), seed 42, on RTX Pro 6000. The
correction makes every candidate feasible (EqL1≈3e-13, IneqL1≈2e-15) regardless
of K, so any merit change is **pure objective improvement**:

| K | ε    | Obj      | IneqL1   | Merit    |
|---|------|----------|----------|----------|
| 1 | 0.00 | -2.7766  | 1.7e-15  | -2.7766  |
| 5   | 0.10 | -3.9924 | 1.8e-15 | -3.9924 |
| 20  | 0.10 | -4.9953 | 1.7e-15 | -4.9953 |
| 100 | 0.05 | -4.8784 | 1.6e-15 | -4.8784 |
| **100** | **0.10** | **-5.5970** | **1.7e-15** | **-5.5970** |
| 100 | 0.30 | -4.9517 | 2.2e-15 | -4.9517 |
| 100 | 1.00 | +8.2872 | 1.4e-14 | +8.2872 (ε too large) |

At 200 epochs, K=1 → K=100 (ε=0.10) appears to **double the objective
improvement** (-2.78 → -5.60). This looked like a large nonconvex-objective
multistart win — **but the controls below show it was overwhelmingly an
UNDERTRAINING artifact.** (At 200 epochs the valid objective was still dropping
fast — it reaches -13.99 by epoch 1500, ~5× better. The 200-epoch NN was simply
nowhere near converged, and perturbation was exploring around a bad point.)

### Controls — mechanism RESOLVED (jobs 14696797 / 14696799, 1500 epochs)

| Setting | K=1 | best K=100 | abs gain | rel gain |
|---------|-----|-----------|----------|----------|
| 200ep nonconvex (orig)  | -2.7766  | -5.5970  | 2.8204 | **101.6%** |
| 1500ep nonconvex (ctrl1)| -13.9913 | -14.1493 | 0.1580 | **1.13%** |
| 1500ep convex (ctrl2)   | -19.7029 | -19.8025 | 0.0996 | **0.51%** |

Reading the three candidate mechanisms:

- **(c) Undertraining — DOMINANT.** Training to convergence (1500 ep) shrinks
  the perturbation gain from 101.6% → 1.13% (an 18× drop in absolute terms,
  2.82 → 0.16). The original "2× win" was ~95% undertraining: perturbation was
  papering over a not-yet-converged NN, exactly like the under-converged-repair
  artifact in FSNet/DC3/Πnet but on the *network* side instead of the *repair*
  side.
- **(b) Non-Euclidean correction — small, real.** Even with a *convex* objective
  at convergence, a 0.51% gain remains. The pinv/ReLU correction is not the
  Euclidean projection, so different perturbed `f` land at slightly different
  feasible points (same class as BP).
- **(a) Nonconvex objective — small, real.** The convex→nonconvex increment at
  convergence is 1.13% − 0.51% ≈ 0.6%. So the genuine nonconvex-objective
  multistart effect exists but is tiny — nothing like the 100% the undertrained
  run suggested.

**Corrected conclusion:** HardNet is NOT a clean large nonconvex-objective win.
At convergence the genuine perturbation gain is ~1% (≈0.5% from the
non-Euclidean correction + ≈0.6% from the nonconvex objective); the rest of the
headline 200-epoch number was undertraining. This adds a **fourth** artifact
source — undertrained network — alongside under-converged repair.

---

## 6. Disconnected-ball multimodal control — RESOLVED (~1% genuine, not 94%)

All five frameworks above had a **convex (connected) feasible set**, so the
solution map is continuous and a converged network needs no ensemble. To find
where ensembling genuinely helps, we built a control with a **disconnected
feasible set**: amortized linear optimization over a union of 4 disjoint balls
(BP's `Disconnected_Ball` geometry).

`min wᵀx s.t. x ∈ ⋃ᵢ Ball(centerᵢ, radiusᵢ)`, conditioning input `[c, w]`
(geometry + objective direction) → `x`. The optimal ball is a **discontinuous**
function of `(c, w)`. Repair = radial bisection from the nearest ball-center
interior point (fully converged, viol ≈ 1e-9). Self-contained solver, trained
unsupervised with ramped feasibility penalty. Script:
`scripts/ensemble/expts/bp/bp_disconnected_sweep.py` (2D, CPU, ~3 min).

**STATUS: RESOLVED.** An earlier draft claimed a confirmed "positive regime —
large gain survives convergence." Controls (`bp_disc_controls.py`,
`bp_disc_convergence.py`) show that was premature — same overclaim pattern as
HardNet. The resolved picture:

**Solid — connected vs disconnected distinction (Control B).** A connected
single-ball (convex) feasible set, same pipeline, is solved at K=1
(optgap 0.0001) and perturbation only *hurts* (0.0001 → 0.0034). So a
perturbation gain genuinely requires the disconnected feasible set; it is not a
pipeline artifact.

**Resolved — the "13% structural floor" was mostly under-optimization.** The
first controls used constant LR and read a "plateau at ~13% wrong-ball." With
proper cosine LR decay and training to convergence (128×4, the best arch), the
K=1 error is far lower and the trajectory flattens near ~5.6%:

| iters (cosine LR) | K=1 optgap | K=1 wrong-ball% | K=100 ε=1 optgap | gain |
|-------------------|-----------|------------------|-------------------|------|
| 60k   | 0.046 | 7.6% | 0.0044 | 0.042 |
| 120k  | 0.032 | 6.8% | 0.0043 | 0.027 |
| 240k  | 0.018 | 5.7% | 0.0041 | 0.014 |
| 480k  | 0.015 | **5.6%** | 0.0041 | **0.011** |

Two facts: (i) more than half the apparent effect was under-optimization — the
"13%" floor fell to ~5.6% once LR decay was used, and the K=100 gain shrank
0.042 → 0.011 as the net converged (same undertraining pattern as HardNet);
(ii) but wrong-ball% genuinely *floors* at ~5.6% (240k→480k: 5.7→5.6, flat),
while optgap keeps creeping down — so there is a **small, real structural
routing floor** (a continuous net cannot perfectly represent the discontinuous
argmin-ball map). The surviving gain at convergence is ≈0.011, i.e. **~1% of the
objective** (|opt| ≈ 1.21) — the same order as BP's ≈0.4% and HardNet's ≈0.6%.

**ε decomposition (128×4, 120k) — usable-ε gain is mostly precision, not
routing:**

| ε | optgap | wrong-ball% |
|---|--------|-------------|
| 0 (K=1) | 0.0664 | 10.7% |
| 0.05 | 0.0278 | 9.8% (barely moves) |
| 0.10 | 0.0207 | 9.1% |
| 0.50 | 0.0073 | 3.2% |
| 1.00 | 0.0046 | 2.2% |

At small ε the optgap falls 58% while wrong-ball is nearly flat — that gain is
within-ball precision (undertraining), **not** ball-switching. True
ball-switching only ramps up at ε ≈ 0.5–1.0, i.e. perturbations comparable to
the whole domain — essentially random multi-start, where the NN is nearly
irrelevant. In 2D, 100 random restarts cover the space, so the headline "94%"
was undertraining + 2D-inflated random search; the convergence-surviving,
genuinely-multimodal part is ~1%.

**Resolved conclusion:** the multimodal/disconnected feasible set is the *most*
real of the convergence-surviving effects — it has a genuine ~5.6% routing
floor that training cannot remove — but the gain is still only ~1% of the
objective, not the 94% headline. The headline was undertraining + random-search
in 2D. So even here, ensembling does not deliver a large convergence-surviving
win; it buys ~1% at K× inference cost.

**Honest conclusion:** the connected-vs-disconnected distinction is real, but a
clean "ensembling delivers a large gain that survives convergence on multimodal
problems" is **not yet demonstrated** — the disconnected K=1 error is still
training-limited, the usable-ε gain is mostly precision, and the routing gain
needs random-search-scale ε in a 2D problem. Pending: the 480k convergence run
(structural floor?) and a higher-dimensional version (does the routing gain
survive d≫2?).

---

## Synthesis

| Framework | Repair operator | Objective | Source of perturbation gain |
|-----------|----------------|-----------|------------------------------|
| FSNet     | L-BFGS (Euclidean, iterative) | convex | **Artifact** — under-converged; fixed by `per_sample_lbfgs=1` |
| DC3       | momentum-SGD on ineq penalty (iterative) | convex | **Artifact** — under-tuned; `corr_lr=1e-3` matches perturb |
| Πnet      | Douglas–Rachford ADMM (Euclidean, iterative) | convex | **Artifact** — under-budgeted; `n_iter_test=500+` matches perturb |
| BP        | radial bisection (non-Euclidean, iterative) | convex | **Real (small)** — repair geometry; ≈0.4% obj, mostly feasibility-tightening |
| HardNet 200ep | one-shot pinv/ReLU correction (closed-form, *non-Euclidean*) | nonconvex | **Mostly artifact** — 101% gain, but ~95% of it was *undertraining* (see controls) |
| HardNet 1500ep (converged) | same | nonconvex | **Real (small)** — 1.1% total: ≈0.5% non-Euclidean correction + ≈0.6% nonconvex objective |
| Disconnected-ball (converged, LR decay) | radial bisection | linear, **disconnected feasible set** | **Real but small (~1%)** — genuine ~5.6% routing floor survives training; the 94% headline was undertraining + 2D random-search (see §6) |

The perturbation gain has **five** sources. The two big-looking ones in the
convex frameworks are both *under-convergence in disguise*; the genuinely large
gain appears only when the feasible set is multimodal:

1. **Under-converged repair** (FSNet, DC3, Πnet — convex obj + iterative
   Euclidean projection). Artifact. Fixing the repair budget is 10–20× cheaper
   and reproduces the numbers. **Confirmed** via budget-sweep controls.

2. **Undertrained network** (HardNet 200ep). Artifact. Training to convergence
   collapses the gain 101% → 1.1% (18× in absolute terms). Same phenomenon as
   (1) but on the *network* axis: perturbation explores around a NN that hasn't
   converged yet. **Confirmed** via the 1500-epoch control.

3. **Direction-dependent repair** (BP radial bisection ≈0.4%; HardNet's
   pinv/ReLU correction ≈0.5%). Genuine but small; survives full convergence
   because the repaired point depends on approach direction.

4. **Nonconvex objective** (HardNet `pᵀsin(y)`, ≈0.6% increment). Genuine but
   tiny — far from the "large multistart win" the undertrained run suggested.

5. **Multimodal (disconnected) feasible set** (disconnected-ball control) →
   **real but small (~1%)** at convergence (§6). The headline ≈94% optgap
   reduction was undertraining + 2D random-search: proper LR-decay training
   halved the apparent routing floor (13%→5.6%) and shrank the K=100 gain
   0.042→0.011. A genuine ~5.6% routing floor *does* survive (continuous net
   can't represent the discontinuous argmin-component map), but the surviving
   objective gain is ~1% — same order as 3–4. The connected-vs-disconnected
   distinction is real (Control B), but does not yield a large convergence-
   surviving win.

Practical rule for L2O ensembling: **the large perturbation/ensemble gains
reported on these benchmarks are convergence artifacts** — under-converged
repair (fix the repair budget) or an undertrained network (train longer, with
LR decay), both far cheaper than ensembling. This held on **all six testbeds**,
*including* the multimodal one: at full convergence every genuine residual is
~1% of the objective. Disconnected/nonconvex feasible sets have the most "real"
residual (a ~5.6% routing floor a continuous net can't remove), but it is still
~1% — not worth K× inference for most uses.

**So: don't ensemble to paper over under-convergence — fix the convergence.**
The diagnostic: push repair budget AND training (with LR decay) to convergence
before measuring any ensemble gain; whatever survives — order ~1% here — is the
real part. Every headline number in this study (60%, 2×, 94%) failed this test.

---

## Recipe: isolating the HardNet mechanism (RUN — results in §5)

Goal: determine whether the HardNet 200ep gain (−2.78 → −5.60) came from the
nonconvex objective, the non-Euclidean correction, or undertraining.
**Result: ~95% undertraining; residual ~1% split between the other two.**

Three controls, each flipping ONE factor, same seed/arch/epochs, same K×ε grid:

1. **Convex-objective control (tests c).** Re-run HardNet-Aff with the objective
   swapped `½yᵀQy + pᵀsin(y)` → `½yᵀQy + pᵀy` (drop the sin; keep Q PSD so the
   objective is convex). Same pinv/ReLU correction.
   - Gain *vanishes* ⇒ the gain was the nonconvex objective (c).
   - Gain *persists* ⇒ it's the non-Euclidean correction (b), not (c).
   Implementation: add a `--convex_obj` flag to `hardnet_verify.py` that
   monkeypatches `data.evaluate`/`get_train_loss` to drop the `sin`.

2. **Euclidean-projection control (tests b).** Re-run with HardNet-**Cvx**
   (`hardnet_cvx.py`, the true QP projection via CvxpyLayer) instead of
   HardNet-Aff, keeping the nonconvex objective.
   - Gain *shrinks toward zero* ⇒ part of the Aff gain was the non-Euclidean
     correction (b).
   - Gain *unchanged* ⇒ (b) is not contributing; consistent with (c).
   (Slow — CvxpyLayer solves a QP per sample — so evaluate on a 128-sample
   subset.)

3. **Convergence control (tests undertraining).** Re-run HardNet-Aff for
   1000–2000 epochs (until valid objective plateaus), then repeat the grid.
   - Gain *shrinks* as training converges ⇒ much of it was exploring around a
     suboptimal NN point (undertraining).
   - Gain *stable* ⇒ undertraining is not the driver.

Decision table after running all three:

| (1) convex-obj gain | (2) Euclidean gain | (3) converged gain | Conclusion |
|---------------------|--------------------|--------------------|------------|
| ~0 | — | stable | Pure nonconvex-objective effect (c) — the clean positive result |
| persists | shrinks | stable | Non-Euclidean correction (b), like BP |
| any | any | shrinks→0 | Undertraining artifact — not a real ensembling win |

Only the top row would justify the headline "nonconvex objective ⇒ genuine
multi-start win." Run order: (3) is cheapest and most likely to undercut the
result, so run it first; then (1); then (2) on a subset.

## Reproducibility

- FSNet/DC3 evidence: scripts under `scripts/ensemble/expts/` (paths in commit
  history; specifically `verify_dc3_unaffected.sh`, `dc3_steps_sweep.sh`,
  `sanity_per_sample.sh`).
- Πnet evidence: `scripts/ensemble/expts/pinet/pinet_verify.{py,sh}`. The repo
  is vendored at `third-party/pinet` (commit pinned by clone; see `git -C
  third-party/pinet log -1`).
- BP evidence: `scripts/ensemble/expts/bp/bp_verify.{py,sh}`. Repo vendored at
  `third-party/Bisection-Projection`.
- HardNet evidence: `scripts/ensemble/expts/hardnet/hardnet_verify.{py,sh}`,
  log `logs/hardnet-verify-14694446.out`. Repo vendored at `third-party/hardnet`.
- Run logs: `logs/pinet-verify-14689578.out` (Πnet QP, seed 42),
  `logs/dc3-steps-sweep-14690284.{out,err}` (DC3 grad_steps budget/lr sweep),
  `logs/dc3-unaff-14687899.out` (DC3 unaffected by L-BFGS per_sample flag).
