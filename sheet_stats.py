"""Barrier statistics on the seed sheets (compute_sheet.py), aggregated over layouts / orderings.

Per sheet (own training loss, points outside the sheet excluded):
  pair barrier    for each pair of lattice-neighbour seeds, the lowest-barrier path on the grid
                  (8-connected, any route inside the sheet) / max(loss at the two seeds)
  escape barrier  for each interior seed, the lowest-barrier path to any seed with lower loss,
                  / loss at the seed (a trap has a high escape barrier)
  low area        share of the sheet within 1 decade of the best seed
Reported as median [IQR] over all pairs / seeds of all given sheets, in decades (log10 ratio).

  python sheet_stats.py --files figures/landscape/fig1/sheet_hex19_o*_test_M2.npz --model M2
"""
import argparse
import glob
import itertools

import numpy as np

from fig1_common import own_loss
from make_m1m2_barrier import minimax_path


def stats(f, m):
    d = dict(np.load(f, allow_pickle=True))
    xs, ys, pos = d["xs"], d["ys"], d["points"]
    c = {k: d[f"plane/{k}"] for k in d["components"]}
    Z = own_loss(m, c)
    inside = np.isfinite(Z)
    if np.nanmin(Z) <= 0:
        Z = Z - np.nanmin(Z) + 1.0
    Zs = np.where(inside, Z, np.inf)
    # each seed -> nearest grid point that lies on the sheet (corner seeds can fall just outside)
    Jg, Ig = np.nonzero(np.isfinite(Z))
    idx = [(int(Jg[k]), int(Ig[k])) for k in
           (np.argmin((xs[Ig] - p[0]) ** 2 + (ys[Jg] - p[1]) ** 2) for p in pos)]
    edges = [(a, b) for a, b in itertools.combinations(range(len(pos)), 2)
             if abs(np.linalg.norm(pos[a] - pos[b]) - 1) < 1e-6]
    pair = [np.log10(max(Zs[p] for p in minimax_path(Zs, idx[a], idx[b])) / max(Zs[idx[a]], Zs[idx[b]]))
            for a, b in edges]
    nbrs = {a: sum(1 for e in edges if a in e) for a in range(len(pos))}
    interior = [a for a in range(len(pos)) if nbrs[a] == 6]
    esc = []
    for a in interior:
        lower = [b for b in range(len(pos)) if Zs[idx[b]] < Zs[idx[a]]]
        if lower:
            esc.append(np.log10(min(max(Zs[p] for p in minimax_path(Zs, idx[a], idx[b])) for b in lower) / Zs[idx[a]]))
    zbest = min(Zs[k] for k in idx)
    low = float((np.log10(Z[inside] / zbest) < 1).mean())
    return pair, esc, low


def fmt(v):
    v = np.asarray(v)
    return f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}, {np.percentile(v, 75):.2f}]" if len(v) else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()
    files = sorted({f for g in args.files for f in glob.glob(g)})
    P, E, A = [], [], []
    for f in files:
        p, e, a = stats(f, args.model)
        P += p
        E += e
        A.append(a)
    print(f"| {args.model} | {len(files)} sheets | pair barrier {fmt(P)} | escape barrier {fmt(E)} | "
          f"low area {np.mean(A):.0%} +- {np.std(A):.0%} |")


if __name__ == "__main__":
    main()
