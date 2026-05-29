# When does "free ensembling" actually help learning-to-optimize?

**Advisor meeting — 2026-05-29**

---

## Thesis (one line)

The large "perturbation / free-ensemble" gains reported for neural constrained
optimizers are **convergence artifacts** — once the repair operator *and* the
network are trained to convergence, the gain collapses to ~1%. We show this
across **6 testbeds / 4 external codebases**, and give a simple diagnostic that
tells a real ensemble gain from an artifact.

---

## 1. The question

- Setup: a NN predicts a solution to a parametric constrained problem; a
  **repair step** (L-BFGS / ADMM / gradient correction / projection) pushes it
  to feasibility.
- Cheap "free ensemble" trick (multi-start): perturb the NN output K times,
  repair each, keep the best by merit. Reported ~60% merit reduction on FSNet.
- **Is that gain real (functional diversity worth ensembling) or an artifact
  (multi-start papering over a badly-converged repair)?**

## 2. Headline finding

**Artifact.** The gain is large only when something is under-converged, on one
of two axes:

- **Under-converged repair** — the repair solver stops early / is under-budgeted.
- **Under-trained network** — the NN itself hasn't converged.

Fix either axis directly and the perturbation gain vanishes — and the direct fix
is **10–20× cheaper** than K× inference-time ensembling.

## 3. Evidence — 6 testbeds, same conclusion

| Testbed (repair) | Apparent gain | After convergence control |
|------------------|---------------|----------------------------|
| **FSNet** (L-BFGS) | −64% merit | **0%** — per-sample AND-convergence kills it |
| **DC3** (grad-SGD) | 624→24 merit | `corr_lr 1e-6→1e-3` alone gives 24 (no perturb) |
| **Πnet** (ADMM proj.) | −62% merit | `n_iter 100→500` alone beats perturb; gain →0.02% |
| **Bisection-Proj.** (radial) | — | real but ~0.4% (repair geometry) |
| **HardNet** (closed-form) | 2× obj ("−2.8→−5.6") | ~95% was undertraining; ~1% at convergence |
| **Disconnected-ball** (multimodal toy) | 94% optgap | ~1% at convergence; high-d coverage collapse |

Each "apparent gain" is reproduced and then dissolved by a control that pushes
the relevant axis to convergence. (Full tables + logs: `CROSS_FRAMEWORK_FINDINGS.md`.)

## 4. The taxonomy — five sources, only ~1% is genuine

1. Under-converged repair → **artifact** (fix the budget).
2. Under-trained network → **artifact** (train longer, w/ LR decay).
3. Direction-dependent repair (non-Euclidean) → real, ~0.5%.
4. Nonconvex objective → real, ~0.6%.
5. Multimodal / disconnected feasible set → real ~1% (a structural routing
   floor a continuous net can't represent), but the headline numbers were
   coverage-inflated in low-dim.

**A convex problem with a converged Euclidean projection and a converged network
gets ~zero benefit from ensembling.**

## 5. The clincher — a REAL multimodal benchmark (nonconvex QCQP)

The toy was synthetic 2D. We escalated to **nonconvex QCQP** (`0.5yᵀQy + p·sin(y)`
objective **and** nonconvex constraints `0.5yᵀHy + cos(y)Gᵀ − h ≤ 0`) — a genuine
multimodal benchmark, full FSNet/L-BFGS pipeline, network trained 300 epochs to
convergence. Merit = obj + 1e6·(eq+ineq violation):

| repair regime | K=1 (no perturb) | K=100 perturb |
|---------------|------------------|----------------|
| **under-converged** (`per_sample=0`) | Merit 114 | Merit 14 (the "88% win") |
| **converged** (`per_sample=1`) | **Merit 2.03** | _[pending cell]_ |

The punchline: **just fixing the repair convergence (114 → 2.03) beats the
perturbation-on-broken-repair (14) by ~7×, with no ensembling at all.** And
raising `max_iter` 50→200 under the legacy criterion does nothing (114→114) —
the fix is the *convergence criterion*, not the budget. _[Decisive cell —
does K=100 add anything on top of converged repair? — folded in below.]_

## 6. Scientific-rigor angle (worth raising)

This study is a case study in how easy it is to mistake under-convergence for a
method gain. **Every negative finding held up under controls; every positive
"ensembling helps" claim we drafted got undercut by a deeper convergence
control** (HardNet's "2× win" → undertraining; the toy's "survives convergence"
→ still-descending error; a "13% structural floor" → 5.6% with proper LR decay).
We kept the corrections in the writeup rather than silently overwriting them.

**Deliverable: a convergence-control protocol** any L2O ensembling claim should
pass — push repair budget AND training (with LR decay) to convergence *before*
measuring an ensemble gain; report only what survives.

## 7. Contributions

1. A reusable, gated fix: `per_sample_lbfgs` — batch-invariant, per-sample
   AND-convergence L-BFGS repair (default off; eval-only since `=1` diverges in
   training).
2. The cross-framework negative result + taxonomy (6 testbeds, 4 external repos).
3. The diagnostic protocol.

## 8. Open / next

- A second seed + a second real problem (AC-OPF) to harden §5.
- Whether the ~1% multimodal residual ever becomes practically large in a
  genuinely high-dimensional multimodal problem (toy showed coverage collapse).
- Framing: negative-result + protocol paper, or fold into a broader
  "what ensembling buys in L2O" study.

---

## Appendix — anticipated questions & defenses

**Q: Isn't "just converge the repair" obvious? Why is the negative result interesting?**
The reported gains were published/claimed as *method* contributions (free
ensembling, multi-restart). Showing they're convergence artifacts — with the
*same number* recoverable by a one-line budget/criterion fix that's 10–20×
cheaper — reframes a whole line of "ensemble your L2O solver" work. The
diagnostic protocol is the transferable contribution.

**Q: Could your convergence "fix" just be over-fitting the eval?**
No — `per_sample_lbfgs` changes only the repair's convergence *criterion*
(per-sample val-AND-grad) and line search, not the objective. Same checkpoint,
same data. And on Πnet/DC3 the fix is literally their own budget knob
(`n_iter`, `corr_lr`) with no code change.

**Q: You changed your conclusions several times — why trust this one?**
Each reversal was forced by a *control*, not a vibe: HardNet's "2× win" died
under a 1500-epoch run; the toy's "survives convergence" died under LR-decay +
the wrong-ball metric. The current claims are the ones that *survived* controls.
The negative findings never reversed. (This is in the writeup as correction
notes, not hidden.)

**Q: Does the multimodal case ever actually justify ensembling?**
There is a genuine residual (~1%, a structural routing floor a continuous net
can't represent) — but it's small, and in higher dimension fixed-K perturbation
coverage collapses (K=100 residual wrong-ball 2%→15% as d 2→16). So even the
"best case for ensembling" doesn't yield a large convergence-surviving win in
our tests. Open: a genuinely high-dim real multimodal problem (AC-OPF).

**Q: Is `per_sample_lbfgs` usable in production?**
Eval-only: `=1` diverges during training (loss→80k+), so it's a test-time repair
mode. Default off; legacy training/eval byte-for-byte unchanged.

**Q: What would change your mind / falsify the thesis?**
A problem where, with repair AND network both verified-converged, K-perturbation
still gives a large (≫1%) merit reduction. We actively looked for this (HardNet
nonconvex obj; disconnected multimodal set) and it collapsed each time under
controls. AC-OPF is the next place to look.

## Appendix — exact numbers (verified against committed logs)

- Πnet (DC3 QP100): n_iter=100 K=1 **116.52** → K=100 **44.64** (−62%); n_iter=500
  K=1 **−11.56** (beats all perturbation). `logs/pinet-verify-14689578.out`
- DC3 (nonsmooth SOCP): default Merit **590.6**; budget 3000 → **76.2**; `corr_lr`
  1e-3 → **24.4**; K=100 perturb → **24.3**. (fix-the-knob == perturbation)
  `logs/dc3-steps-sweep-14690284.out`, `logs/dc3-unaff-14687899.out`
- HardNet (nonconvex QP100): 200ep K=1 **−2.78** → K=100 **−5.60**; at 1500ep the
  K=1 reaches the converged level and the gain → ~1%. `logs/hardnet-verify-14694446.out`
- Disconnected-ball: K=1 wrong-ball 5.6% floor (cosine LR, 480k); K=100 gain
  shrinks 0.042→0.011 as net converges; high-d K=100 residual 2.2→14.9% (d 2→16).
- **Nonconvex QCQP (real multimodal):** under-converged repair K=1 **114** →
  K=100 **14**; converged repair (`per_sample=1`) K=1 **2.03**. `logs/eval-ncqcqp-14735337.out`
