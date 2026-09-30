"""Merge row shards written by compute_landscape_compare.py --shard k/n into one .npz.

  python merge_shards.py --out figures/landscape/fig1/seedplane_aligned_test_M3f.npz --n 4
reads <out stem>_shard{k}.npz for k < n.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    args = ap.parse_args()
    stem = args.out[:-4] if args.out.endswith(".npz") else args.out
    parts = [dict(np.load(f"{stem}_shard{k}.npz", allow_pickle=True)) for k in range(args.n)]
    out = dict(parts[0])
    for key in [k for k in out if k.startswith("plane/")]:
        Z = out[key].copy()
        for p in parts[1:]:
            Z = np.where(np.isnan(Z), p[key], Z)
        out[key] = Z
        if np.isnan(Z).any():
            print(key, "NaN points after merge:", int(np.isnan(Z).sum()))
    np.savez(args.out, **out)
    print(f"merged {args.n} shards -> {args.out}")


if __name__ == "__main__":
    main()
