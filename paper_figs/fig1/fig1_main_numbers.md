# Fig. 1 numbers (held-out, own training loss, decades = log10 ratio)

## Seed-triple typicality

Median straight-line edge barrier of each triple; in brackets its percentile among the 18 edges of the method's 10-model sheet.

| triple | M1 | M4 | M2 | M3f | summed abs(pct - 50) |
|---|---:|---:|---:|---:|---:|
| (0, 1, 2) | 0.58 (0th) | 1.89 (78th) | 2.96 (100th) | 0.33 (6th) | 172 |
| (3, 4, 5) | 0.68 (44th) | 2.09 (100th) | 2.37 (83th) | 2.62 (33th) | 106 |
| (6, 7, 8) | 0.69 (56th) | 1.93 (83th) | 2.67 (100th) | 2.59 (33th) | 106 |

Chosen for the main figure: seeds (3, 4, 5).

## Along the line s0 - s1 (barrier relative to the higher of the two networks)

| triple | method | barrier between s0 and s1 |
|---|---|---:|
| (0, 1, 2) | M1 | 4x |
| (0, 1, 2) | M4 | 73x |
| (0, 1, 2) | M2 | 1,507x |
| (0, 1, 2) | M3f | 2x |
| (3, 4, 5) | M1 | 5x |
| (3, 4, 5) | M4 | 127x |
| (3, 4, 5) | M2 | 548x |
| (3, 4, 5) | M3f | 312x |
| (6, 7, 8) | M1 | 5x |
| (6, 7, 8) | M4 | 146x |
| (6, 7, 8) | M2 | 1,236x |
| (6, 7, 8) | M3f | 375x |

## Hard FS at rho = 1 and 0.8 on the same triples (barrier s0 to s1)

| triple | method | s0, s1 status | barrier between s0 and s1 |
|---|---|---|---:|
| (0, 1, 2) | M3r1 | good, collapsed | 7x |
| (0, 1, 2) | M3r08 | good, good | 1x |
| (3, 4, 5) | M3r1 | collapsed, good | 1x |
| (3, 4, 5) | M3r08 | good, good | 1x |
| (6, 7, 8) | M3r1 | good, good | 1x |
| (6, 7, 8) | M3r08 | collapsed, collapsed | 1x |
