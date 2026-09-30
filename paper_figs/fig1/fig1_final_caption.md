# Fig. 1 caption draft

Loss landscapes of the own training loss around 19 independently trained networks per method
(nonsmooth nonconvex SOCP, L1 penalty). Each panel is a piecewise-planar sheet: the 19 trained
networks, permutation-aligned to one of them (weight matching), are placed on a triangular
lattice, and inside each lattice triangle the surface is the exact plane through its three
networks. Colour: log10 of the loss relative to the best network on the sheet (held-out test
instances), capped at 3 decades. Red dots: trained networks. Pair barrier: lowest-barrier path
between neighbouring networks, relative to their loss; escape barrier: lowest path from an
interior network to any better one. Medians over 3 random assignments of networks to lattice sites.

| model | split | pair barrier (median) | escape barrier (median) |
|---|---|---:|---:|
| M1 | test | 4.7x (126 pairs) | 4.5x (20 seeds) |
| M1 | train | 8.8x (42 pairs) | 10.0x (6 seeds) |
| M2 | test | 72.6x (126 pairs) | 133.9x (18 seeds) |
| M2 | train | 72.8x (42 pairs) | 142.6x (6 seeds) |
| M2m | test | 42.9x (126 pairs) | 48.7x (18 seeds) |
| M2m | train | 47.1x (42 pairs) | 63.2x (6 seeds) |
| M4 | test | 37.7x (126 pairs) | 43.4x (18 seeds) |
| M4 | train | 39.9x (42 pairs) | 47.9x (6 seeds) |
