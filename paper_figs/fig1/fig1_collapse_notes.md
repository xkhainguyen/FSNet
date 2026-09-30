# Collapse panel (fig1_collapse.png): hard FS at rho = 1 and 0.8

Planes through permutation-aligned seeds, s0 = lowest-index good seed, s1 = lowest-index collapsed
seed, s2 = next good (A) or next collapsed (B). Collapsed = merit after FS (500 it) > 10
(m3_candidates.py): rho 1 seeds 1, 3, 8 (3/10); rho 0.8 seeds 2, 5, 6, 7, 8 (5/10).
Rows: own training loss (3D, top view; 100 Huber(FS(y) - y*) + rho * raw L1 violation), merit after
FS (top view), own loss along s0 -> s1. Colour: log10 relative to the method's best of its 10 trained
networks (seed_losses.py), cap 4. The same variants in the main-figure layout: fig1_main_all*.png.

Findings
- Good seeds sit in a narrow low valley; every collapsed seed sits on a vast flat plateau
  (86-97% of each plane within 1.2-1.4 decades, about 20x above the good valley). Along s0 -> s1 the
  loss climbs out of the valley and then stays flat through the collapsed seed and beyond.
- On the plateau the raw output is far from feasible (rho * L1 violation ~1e3 vs 25), FS moves it a
  lot (||FS(y) - y||^2 ~1.9e3 vs 2-3) but does not reach feasibility (violation after FS 5-15 vs
  1e-3), and the objective after FS is ~75 vs 0.6.
- Mechanism: the output sigmoid is saturated. All 8 collapsed seeds have 100% of outputs within
  1e-3 of a variable bound (sigmoid slope 3e-5 to 1e-4); all 12 good seeds, and the rho = 10 SL and
  FSNet-style hard-FS networks, have 0% (slope 0.23) (m3_saturation.py). Saturated outputs are
  pinned at the bounds, so the loss is flat over a wide region and the gradient through the
  sigmoid vanishes: training stalls there. This is a dead plateau, not a local-minimum trap.
- Hypothesis (not yet tested on training curves): with a weak raw-output penalty (rho <= 1) and no
  distance term, the gradient through FS does not keep the raw output near the feasible set, so it
  can drift into the saturated region; the FSNet recipe (gated squared penalty + 5 ||FS(y) - y||^2)
  keeps it near FS(y) and never saturated (0/10 collapsed).
