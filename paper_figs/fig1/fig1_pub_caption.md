# Fig. 1 caption draft (fig1_pub.pdf / .png, make_fig1_pub.py)

**Training-loss landscapes on the nonsmooth nonconvex SOCP (100-50-50), held-out instances.**
Each column is one training method: supervised learning (SL) or self-supervised learning (SSL) with
a soft L1 constraint penalty of weight rho, or SL with a hard feasibility-seeking (FS) layer in the
loop and a raw-output penalty rho = 1. Each panel shows the method's own training loss on the plane
through three independently trained networks (seeds 0, 1, 2), permutation-aligned by weight
matching. Top: 3D surface; middle: top view; bottom: the loss along the line through s0 and s1
(dashed in the middle row; grey band between the two networks). Colour and height: log10 of the loss
relative to L*, the method's best of its 10 trained networks, on one scale for all methods (capped
at 4 decades in the top two rows). Red dots: trained weights (for hard FS, rho = 1, the network at s1 is a collapsed one: merit after FS > 10); star: the minimum on the plane of the merit, objective + 1e5 x constraint violation (after FS for
hard FS), i.e. the point that is best for the task the methods are judged on. Barrier: the largest rise of the loss above the straight line between its values at
s0 and s1 (Frankle et al., 2020).

- SL, rho = 10: one wide, shallow basin; a 4x barrier between the networks.
- SSL, rho = 10: separate pits between smooth ridges (82x).
- SL, rho = 1e5: needle-thin pits cut into a high, sloped landscape, the highest barrier (1631x).
- SL, hard FS, rho = 1: the good networks lie in a thin low trough with erratic spikes along its
  edges; the collapsed network sits on a flat plateau (about 20x above L*) covering most of the plane.
  From it the loss stays flat before dropping into the trough: a gradient-free plateau (the output
  sigmoid is saturated at the variable bounds). Its 49x barrier comes mostly from the erratic
  spikes at the trough edge.

## Details for the text or appendix

- Barrier definition: the standard linear-interpolation barrier. Relative to the higher endpoint
  instead (as in fig1_main*.png) the four values are 4x, 73x, 1507x and 7x: the soft methods barely
  change, but for hard FS rho = 1 that definition hides the climb from the good network because the
  collapsed endpoint is itself high. The hard-FS rho = 1 barrier is not robust across triples
  (49x, 6x, 1x for (0, 1, 2), (3, 4, 5), (6, 7, 8)). The robust measure of its failure is the flat
  plateau: the share of the plane more than 10x above L* with a slope below 0.1 decades per unit of
  ||theta_1 - theta_0|| is 60%, 58%, 64% on the three triples, against at most 6% for every other method
  on every triple (make_fig1_pub.plateau).
- The seed triple (0, 1, 2) was chosen for display. By the typicality rule of make_fig1_main.py it is
  the least typical of the three triples for the soft methods: rho = 10 sits at the 0th and rho = 1e5
  at the 100th percentile of their 10-network edge barriers. The other triples give 5x, 154-165x and
  571-1306x (standard barrier) for rho = 10, SSL and rho = 1e5, so the ordering is the same but this
  triple shows rho = 1e5 at its largest.
- Hard FS with rho = 1 (no gate, no distance term, 1000 epochs) collapses on 3 of 10 seeds (1, 3, 8);
  all collapsed networks have 100% of their sigmoid outputs within 1e-3 of a bound (slope 3e-5 to 1e-4
  vs 0.23; m3_saturation.py). The FSNet recipe (gated squared penalty, distance term 5, 300 epochs)
  collapses on none (fig1_main.png, fig1_main_all*.png).
- Loss: each method's own training objective with the trainer's weights, FS via the evaluation
  solver (L-BFGS, 50 iterations, tolerance 1e-9; training used 1e-7). L* is evaluated exactly at the
  weights (seed_losses.py); rho = 1e5 pits are narrower than the grid, so its networks plot 0.1 to 0.25
  decades above zero and its barriers are lower bounds.
- Plane: Gram-Schmidt basis of the three networks, units of ||theta_1 - theta_0||, 151 x 151 grid,
  shown on x in [-1, 2], y in [-0.8, 1.7]; 1000 held-out instances per point.
- Linear vs curved: along learned curved paths the rho = 10, SSL and hard-FS networks are nearly
  connected, while rho = 1e5 keeps a median 9x barrier (fig1_sheet10_curved.png).
- Normalization floor: 44% of the rho = 10 loss at its networks is the label term, so log(L / L*)
  compresses its landscape; in constraint violation alone its segment rises 8x vs 550x for rho = 1e5
  (seeds 3, 4, 5).

## Variant with rho = 1 for the soft-penalty SL and SSL columns (fig1_pub_rho1.pdf / .png)

Columns: SL soft penalty rho = 1, SSL soft penalty rho = 1, SL soft penalty rho = 1e5, SL hard FS rho = 1
(`python make_fig1_pub.py rho1`). Same plane, seeds, crop and normalization as the rho = 10 figure.

- SL rho = 1: one wide, shallow basin. Across the three seed triples the barrier is 1x, 4x, 3x (rho = 10:
  4x, 5x, 5x), the plane stays within 1.1 decades of L* (rho = 10: 2.1) and 66% of it is within 10x of L*
  (rho = 10: 23%). Cost: its L1 constraint violation is 82 against 4-6 at rho = 10, and its merit is 15x
  worse (8.2e6 vs 5.6e5): the nicest landscape is the least feasible solution.
- SSL rho = 1: the trained networks fall into two groups. Six of ten seeds reach an objective near -4.0
  (the lowest of any method) with L1 violation about 0.9-1.0; the others sit at objectives between -1 and
  +7.5. Their own loss (objective + 1 x violation) is therefore negative for six seeds, so log(L / L*)
  is undefined. The panel shows log10(L - L* + 1) with L* the best trained network's loss; barriers
  on that scale (41x, 34x, 18x per triple, marked with a dagger) are not comparable with the ratios.
  In every triple one network is a visibly worse local minimum than the other two.
- SL and SSL share the same rho number but not the same balance: at the trained networks the penalty is
  about 62% of the SL rho = 1 loss (objective term 100 x Huber ~ 50) and 87-99% of the SSL loss (its
  objective term is near zero), so SSL at rho = 1 is penalty-dominated like SL at a much larger rho.

- Merit star (both variants): it lies at or next to a trained network in every panel (0.0 to 0.4 in units of
  ||theta_1 - theta_0||), so the methods reach the right basins but not good merit values; the merit levels
  are in fig1_mismatch.png (median merit at the 10 networks: SL rho = 1 8.1e6, SSL rho = 1 9.5e4, SL
  rho = 1e5 4.5e4, hard FS rho = 1 1.2e2; SL rho = 10 5.6e5, SSL rho = 10 7.4e4). fig1_mismatch is now
  optional (supplementary): the star is the only piece of it needed in the main landscape figure.
