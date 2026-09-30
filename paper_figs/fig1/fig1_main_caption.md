# Fig. 1 caption draft (fig1_main.png)

**Training-loss landscapes on the nonsmooth nonconvex SOCP (100-50-50), held-out instances.**
Each column is one training method; each panel shows that method's own training loss on the plane
through three independently trained networks (seeds 3, 4, 5), permutation-aligned by weight
matching. Top: 3D surface; middle: top view; bottom: the loss along the straight line through two
of the networks (dashed in the middle row; the grey band is the segment between them). Colour and
height: log10 of the loss relative to the method's best of its 10 trained networks (evaluated
exactly at the weights), one scale for all methods,
capped at 4 decades in the top two rows (uncapped in the bottom row; triangles mark values above 6
decades). Red dots: trained networks. Columns follow the narrative (soft penalty, SSL, large rho, hard FS),
not the barrier size. Barriers are the highest loss on the segment relative to the
higher of its two endpoints.

- SL, soft penalty, rho = 10: a wide, shallow basin; the networks are separated by a 5x barrier.
- SSL, soft penalty, rho = 10: isolated pits separated by smooth ridges (127x).
- SL, soft penalty, rho = 1e5: narrower pits and the highest barrier (548x).
- SL, hard FS (feasibility-seeking layer in the loop, FSNet-style gated penalty): a low region
  bounded by vertical cliffs where the gated penalty on the raw output switches on (312x between
  the networks here; 2x to 375x across seed triples) and, further out, an erratic plateau where the
  feasibility solver diverges.

## Details for the text or appendix

- Plane: spanned by the three networks (Gram-Schmidt basis; s0 at (0, 0), s1 at (1, 0), units of
  ||theta_1 - theta_0||), 151 x 151 grid with a margin of 1 around the three networks
  (x in [-1, 2], y in [-1, 1.88]), 1000 held-out instances per point; shown cropped to
  y in [-0.8, 1.7] (the cropped strip holds no value below 1.7 decades for any method).
- Loss: each method's own training objective with the trainer's weights (Huber label term with
  weight 100 for SL; objective for SSL; L1 penalty with rho for the soft methods; for hard FS
  100 * Huber(FS(y) - y*) + 5 * ||FS(y) - y||^2 + 10 * squared raw violation, the last term only when
  the batch-mean squared eq or ineq violation is >= 1e3, applied per evaluation batch of 500).
  The FS layer uses the evaluation solver (L-BFGS, 50 iterations, tolerance 1e-9; training used
  tolerance 1e-7 and batches of 512).
- Training budgets differ: hard FS follows the FSNet recipe (300 epochs, lr 1e-4); the soft methods
  use 3000 epochs with lr 3e-4 (rho = 10), 1e-4 (rho = 1e5, its best lr) and 1e-3 (SSL). An lr-matched
  control of rho = 1e5 (lr 3e-4, "M2m") keeps a large barrier (43x vs 73x median lowest-path pair
  barrier on the 19-network sheets).
- Seed triple: chosen by a fixed rule, the triple whose median straight-line edge barrier is closest
  to each method's median over the 18 edges of its 10-network sheet (summed percentile distance from
  50). (3, 4, 5) and (6, 7, 8) tie exactly (106 vs 172 for (0, 1, 2)); the tie is broken by index.
  The rule was fixed after the three planes had been computed and viewed, before the final layout.
  (6, 7, 8) gives the same ordering (5x, 146x, 1236x, 375x). In (0, 1, 2), the least typical triple,
  the hard-FS pair lies inside one gate-off region (2x) while rho = 1e5 is at 1507x
  (fig1_main_plane.png, fig1_main_plane_t678.png, fig1_main_numbers.md).
- Normalization: dividing by the best network's loss compresses methods whose loss has a large
  floor. At the rho = 10 networks 44% of the loss is the Huber label term; at rho = 1e5 almost none.
  In constraint violation alone, the rho = 10 segment rises 0.88 decades (8x) and the rho = 1e5
  segment 2.74 decades (550x); evaluating the rho = 10 networks under the rho = 1e5 loss gives 8x,
  and the rho = 1e5 networks under the rho = 10 loss 26x. So the rho = 1e5 networks do sit in much
  sharper wells, but part of the 5x vs 548x contrast is the loss floor.
- Hard FS: the cliff is the penalty gate (a designed discontinuity of the FSNet-style loss), not an
  emergent property of the FS layer; at the networks the loss is 99% label error, at the triangle
  centre 99% gated penalty. Outside the cliff the L-BFGS feasibility solve diverges (the FS output
  moves up to about 6e7 from the raw prediction, i.e. ||FS(y) - y||^2 up to 3e15; 34% of the plane has
  ||FS(y) - y||^2 > 1e3), because its backtracking line search takes the step even when no trial is
  accepted and its curvature scaling is unguarded. Along s0 - s1 the
  hard-FS raw-output violation rises 1.34 decades vs 0.88 for rho = 10.
- Resolution: rho = 1e5 pits are narrower than the grid (its networks plot 0.1 to 0.25 decades above
  zero for that reason); one grid step from a network already costs
  0.7 to 0.9 decades here (0.5 to 1.5 across triples), so its pit widths are grid-limited and its barriers are lower bounds.
- Linear vs curved paths: these are planar views. Along learned curved paths (Garipov et al., 2018)
  the rho = 10, SSL and hard-FS networks are nearly connected (median 1x to 2x; SSL up to 3x), while
  rho = 1e5 keeps a median 9x barrier (up to 16x) (fig1_sheet10_curved.png): large rho is the only case whose minima stay
  separated. The planar picture shows how the loss behaves between and around trained networks,
  not that no low path exists.

## Extended figure (fig1_main_all*.png): hard FS at rho = 1 and 0.8 added

Same layout and seed triples, plus hard FS with a plain L1 raw-output penalty (rho = 1 or 0.8, no
gate, no distance term, 1000 epochs). These runs collapse on 3/10 and 5/10 seeds (merit after FS > 10;
black X). The good networks share a thin low trough (1x between two good networks); the rest of the
plane, where the collapsed networks sit, is a flat plateau about 1.2 to 1.4 decades above the best
network, with erratic spikes at the trough edges. Along a line from a collapsed to a good network
the loss stays flat before dropping into the trough, so the printed "1x" barrier there means no
barrier but a flat, gradient-free stretch. Every collapsed network has 100% of its sigmoid outputs
saturated at the variable bounds (slope 3e-5 to 1e-4 vs 0.23 for good networks; m3_saturation.py):
a dead plateau, not a local-minimum trap. With all three networks of a triple collapsed
(rho = 0.8, seeds 6, 7, 8) the whole plane is that plateau.
