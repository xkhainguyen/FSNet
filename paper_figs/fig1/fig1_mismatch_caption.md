# Objective-mismatch figure caption draft (fig1_mismatch.pdf / .png, make_fig1_mismatch.py)

**What each method minimizes is not what it is judged on.** Columns: SL with a soft L1 penalty, rho = 1;
SSL with a soft L1 penalty, rho = 1; SL with a soft L1 penalty, rho = 1e5; SL with the hard feasibility-seeking
(FS) layer in the loop, rho = 1. Each panel is the plane through three independently trained, permutation-aligned
networks (seeds 0, 1, 2; red dots), on held-out instances. Top: the method's own training loss, as in Fig. 1
(log10 of the loss relative to its best trained network; the SSL loss goes negative, so that column shows
log10(L - L* + 1)). Middle: the merit, objective + 1e5 x L1 constraint violation (after FS for hard FS), on the
same plane, one absolute log10 scale for all columns; the star marks the merit minimum on the plane. Bottom: the
merit at each method's 10 trained networks (symlog axis: linear between -10 and 10, logarithmic beyond; the number under each strip is its median). The solid and dashed grey lines are references
on 100 held-out instances: the merit of the SL labels (-2.4) and of the best of 20 random-start IPOPT solves per
instance (-4.3).

## What it shows

- The landscape that looks best to train is not the one that reaches good merit. SL at rho = 1 has the
  widest, shallowest training basin (Fig. 1) and the worst merit at its networks (median 8.1e6, 15x worse than
  rho = 10, against references of about -3); its merit landscape is uniformly high.
- SSL (9.5e4) and SL at rho = 1e5 (4.5e4) have much lower merit, and there the training loss is nearly the merit
  (for rho = 1e5 the two coincide: Spearman rank correlation between training loss and merit over the 10
  seeds is +1.00).
- Only hard FS reaches merit within a few units of the references, and only in a thin trough; the
  merit landscape outside it is as high as for the other methods. 3 of its 10 networks (the collapsed ones,
  merit 1.6e5 to 1.4e6) are far worse than the good seven (2.7 to 4.1e2, three of them at about 3).
- In none of the four planes is the merit minimum far from a trained network (0.0 to 0.4 in units of
  ||theta_1 - theta_0||), so the mismatch is mainly in the level, not the location: no trained network
  reaches the merit of the labels, let alone the best achievable.

## Per-method numbers (10 networks each)

| method | merit range | Spearman(training loss, merit) | best network by training loss is merit rank |
|---|---|---:|---:|
| SL, rho = 10 | 4.0e5 to 6.1e5 | +1.00 | 1 of 10 |
| SL, rho = 1 | 8.0e6 to 8.3e6 | +0.90 | 1 of 10 |
| SSL, rho = 10 | 6.7e4 to 8.2e4 | -0.35 | 9 of 10 |
| SSL, rho = 1 | 7.8e4 to 1.2e5 | +0.27 | 3 of 10 |
| SL, rho = 1e5 | 4.1e4 to 5.1e4 | +1.00 | 1 of 10 |
| SL, hard FS, rho = 1 | 2.7 to 1.4e6 | +0.44 | 7 of 10 |

For SSL and hard FS the training loss does not even rank the networks by merit.

## The SL labels are local optima (label_multistart.py)

The labels are single IPOPT solves from the zero start of a nonconvex problem (99% of instances reproduce
the label exactly from zero). On 100 held-out instances, 20 IPOPT starts each: 75 of 100 have a feasible
point better than the label (mean objective -2.42 for the labels vs -4.31 for the best of 20; median improvement
0.85, max 14.6), and the 20 starts land on 20 distinct local optima for the median instance. SL therefore
regresses onto an arbitrary local optimum per instance. Caveat: the trained SSL and SL networks do not
reach these objectives while feasible (their merit is 4e4 and up).

## Notes

- Merit in the top plane of a method with the FS layer is evaluated after the evaluation solver (L-BFGS, 50
  iterations); beyond the trough it diverges (values up to 1e18 are clipped at 1e8 in the colours).
- Plot the ranking table from seed_losses.json; references from label_multistart.json.
