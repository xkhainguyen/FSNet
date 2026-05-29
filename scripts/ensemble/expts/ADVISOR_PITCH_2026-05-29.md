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
