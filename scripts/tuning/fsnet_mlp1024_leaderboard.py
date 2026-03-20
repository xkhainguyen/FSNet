#!/usr/bin/env python3
"""Aggregate Wave-1 FSNet MLP1024 runs into a 4-seed leaderboard.

Primary rank: lower mean opt_gap_mean (%).
Tie-breaker: lower std opt_gap_mean (%).
"""

import argparse
import csv
import os
from typing import Dict, List, Optional

import numpy as np
import yaml


class Row:
    def __init__(
        self,
        cfg_id,
        seed,
        lr,
        dropout,
        dist_weight,
        save_dir,
        run_log,
        val_tol=None,
        decay_tol_step=None,
        memory_size=None,
        max_diff_iter=None,
    ):
        self.cfg_id = cfg_id
        self.seed = seed
        self.lr = lr
        self.dropout = dropout
        self.dist_weight = dist_weight
        self.save_dir = save_dir
        self.run_log = run_log
        self.val_tol = val_tol
        self.decay_tol_step = decay_tol_step
        self.memory_size = memory_size
        self.max_diff_iter = max_diff_iter


class EvalPoint:
    def __init__(
        self,
        cfg_id,
        seed,
        lr,
        dropout,
        dist_weight,
        opt_gap_mean,
        opt_gap_std,
        opt_gap_max,
        eq_l1_mean,
        ineq_l1_mean,
        save_dir,
        val_tol=None,
        decay_tol_step=None,
        memory_size=None,
        max_diff_iter=None,
    ):
        self.cfg_id = cfg_id
        self.seed = seed
        self.lr = lr
        self.dropout = dropout
        self.dist_weight = dist_weight
        self.opt_gap_mean = opt_gap_mean
        self.opt_gap_std = opt_gap_std
        self.opt_gap_max = opt_gap_max
        self.eq_l1_mean = eq_l1_mean
        self.ineq_l1_mean = ineq_l1_mean
        self.save_dir = save_dir
        self.val_tol = val_tol
        self.decay_tol_step = decay_tol_step
        self.memory_size = memory_size
        self.max_diff_iter = max_diff_iter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FSNet MLP1024 leaderboard from manifest")
    p.add_argument(
        "--manifest",
        default="logs/fsnet_tuning_wave1/wave1_manifest.tsv",
        help="TSV created by scripts/tuning/fsnet_mlp1024_wave1.slurm.sh",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out-csv", default="logs/fsnet_tuning_wave1/wave1_leaderboard.csv")
    return p.parse_args()


def load_manifest(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(
                Row(
                    cfg_id=r["cfg_id"],
                    seed=int(r["seed"]),
                    lr=float(r["lr"]),
                    dropout=float(r["dropout"]),
                    dist_weight=float(r["dist_weight"]),
                    save_dir=r["save_dir"],
                    run_log=r["run_log"],
                    val_tol=float(r["val_tol"]) if r.get("val_tol") else None,
                    decay_tol_step=int(r["decay_tol_step"]) if r.get("decay_tol_step") else None,
                    memory_size=int(r["memory_size"]) if r.get("memory_size") else None,
                    max_diff_iter=int(r["max_diff_iter"]) if r.get("max_diff_iter") else None,
                )
            )
    return rows


def load_summary_metrics(save_dir: str, batch_size: int) -> Optional[dict]:
    path = os.path.join(save_dir, "test_summary.yaml")
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        text = f.read()

    # Older summaries may serialize torch version with a python-specific tag.
    # Strip this field so parsing works even when torch is unavailable.
    cleaned = []
    skip_next = False
    for line in text.splitlines():
        if line.startswith("pytorch_version: !!python/object/new:torch.torch_version.TorchVersion"):
            skip_next = True
            continue
        if skip_next and line.lstrip().startswith("- "):
            skip_next = False
            continue
        skip_next = False
        cleaned.append(line)
    text = "\n".join(cleaned) + "\n"

    try:
        summary = yaml.safe_load(text) or {}
    except yaml.constructor.ConstructorError:
        # Legacy files may include python-specific tags.
        if hasattr(yaml, "unsafe_load"):
            summary = yaml.unsafe_load(text) or {}
        else:
            summary = yaml.load(text, Loader=yaml.Loader) or {}

    test_block = summary.get("test", {})
    k = int(batch_size)
    metrics = test_block.get(k, test_block.get(str(k)))
    if not isinstance(metrics, dict) or "error" in metrics:
        return None
    return metrics


def build_points(rows: List[Row], batch_size: int) -> List[EvalPoint]:
    points: List[EvalPoint] = []
    for r in rows:
        m = load_summary_metrics(r.save_dir, batch_size)
        if m is None:
            continue
        if "opt_gap_mean" not in m:
            continue

        points.append(
            EvalPoint(
                cfg_id=r.cfg_id,
                seed=r.seed,
                lr=r.lr,
                dropout=r.dropout,
                dist_weight=r.dist_weight,
                opt_gap_mean=float(m["opt_gap_mean"]),
                opt_gap_std=float(m.get("opt_gap_std")) if m.get("opt_gap_std") is not None else None,
                opt_gap_max=float(m.get("opt_gap_max")) if m.get("opt_gap_max") is not None else None,
                eq_l1_mean=float(m.get("eq_violation_l1_mean")) if m.get("eq_violation_l1_mean") is not None else None,
                ineq_l1_mean=float(m.get("ineq_violation_l1_mean")) if m.get("ineq_violation_l1_mean") is not None else None,
                save_dir=r.save_dir,
                val_tol=r.val_tol,
                decay_tol_step=r.decay_tol_step,
                memory_size=r.memory_size,
                max_diff_iter=r.max_diff_iter,
            )
        )
    return points


def aggregate(points: List[EvalPoint]) -> List[dict]:
    by_cfg: Dict[str, List[EvalPoint]] = {}
    for p in points:
        by_cfg.setdefault(p.cfg_id, []).append(p)

    rows: List[dict] = []
    for cfg_id, pts in by_cfg.items():
        opt = np.array([x.opt_gap_mean for x in pts], dtype=float)
        eq = np.array([x.eq_l1_mean for x in pts if x.eq_l1_mean is not None], dtype=float)
        ineq = np.array([x.ineq_l1_mean for x in pts if x.ineq_l1_mean is not None], dtype=float)
        max_gap = np.array([x.opt_gap_max for x in pts if x.opt_gap_max is not None], dtype=float)

        base = pts[0]
        present = sorted(x.seed for x in pts)
        missing = [s for s in [0, 1, 2, 3] if s not in present]

        rows.append(
            {
                "cfg_id": cfg_id,
                "lr": base.lr,
                "dropout": base.dropout,
                "dist_weight": base.dist_weight,
                "val_tol": base.val_tol,
                "decay_tol_step": base.decay_tol_step,
                "memory_size": base.memory_size,
                "max_diff_iter": base.max_diff_iter,
                "n_seeds": len(pts),
                "seed_list": present,
                "missing_seeds": missing,
                "opt_gap_mean_mean": float(np.mean(opt)),
                "opt_gap_mean_std": float(np.std(opt)),
                "opt_gap_mean_min": float(np.min(opt)),
                "opt_gap_mean_max": float(np.max(opt)),
                "opt_gap_max_mean": float(np.mean(max_gap)) if max_gap.size else np.nan,
                "eq_l1_mean_mean": float(np.mean(eq)) if eq.size else np.nan,
                "ineq_l1_mean_mean": float(np.mean(ineq)) if ineq.size else np.nan,
            }
        )

    rows.sort(key=lambda r: (r["opt_gap_mean_mean"], r["opt_gap_mean_std"]))
    return rows


def print_table(rows: List[dict]) -> None:
    # Detect if wave-2 knobs are present
    has_knobs = any(r.get("val_tol") is not None for r in rows)
    if has_knobs:
        header = (
            f"{'rank':>4} {'cfg_id':>12} {'lr':>8} {'drop':>6} {'dist':>6} "
            f"{'val_tol':>8} {'decay':>6} {'mem':>4} {'diff':>5} "
            f"{'n':>3} {'opt_mean':>12} {'opt_std':>12} {'missing':>10}"
        )
        print(header)
        print("-" * len(header))
        for i, r in enumerate(rows, 1):
            miss = "-" if not r["missing_seeds"] else ",".join(map(str, r["missing_seeds"]))
            vt = r['val_tol'] if r['val_tol'] is not None else float('nan')
            dt = r['decay_tol_step'] if r['decay_tol_step'] is not None else 0
            ms = r['memory_size'] if r['memory_size'] is not None else 0
            mdi = r['max_diff_iter'] if r['max_diff_iter'] is not None else 0
            print(
                f"{i:4d} {r['cfg_id']:>12} {r['lr']:8.1e} {r['dropout']:6.2f} {r['dist_weight']:6.2f} "
                f"{vt:8.0e} {dt:6d} {ms:4d} {mdi:5d} "
                f"{r['n_seeds']:3d} {r['opt_gap_mean_mean']:12.4f} {r['opt_gap_mean_std']:12.4f} {miss:>10}"
            )
    else:
        header = (
            f"{'rank':>4} {'cfg_id':>8} {'lr':>8} {'dropout':>8} {'dist_w':>8} "
            f"{'n':>3} {'opt_mean':>12} {'opt_std':>12} {'opt_min':>12} {'opt_max':>12} {'missing':>10}"
        )
        print(header)
        print("-" * len(header))
        for i, r in enumerate(rows, 1):
            miss = "-" if not r["missing_seeds"] else ",".join(map(str, r["missing_seeds"]))
            print(
                f"{i:4d} {r['cfg_id']:>8} {r['lr']:8.1e} {r['dropout']:8.2f} {r['dist_weight']:8.2f} "
                f"{r['n_seeds']:3d} {r['opt_gap_mean_mean']:12.4f} {r['opt_gap_mean_std']:12.4f} "
                f"{r['opt_gap_mean_min']:12.4f} {r['opt_gap_mean_max']:12.4f} {miss:>10}"
            )


def write_csv(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "cfg_id",
        "lr",
        "dropout",
        "dist_weight",
        "val_tol",
        "decay_tol_step",
        "memory_size",
        "max_diff_iter",
        "n_seeds",
        "seed_list",
        "missing_seeds",
        "opt_gap_mean_mean",
        "opt_gap_mean_std",
        "opt_gap_mean_min",
        "opt_gap_mean_max",
        "opt_gap_max_mean",
        "eq_l1_mean_mean",
        "ineq_l1_mean_mean",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["seed_list"] = "|".join(map(str, r["seed_list"]))
            out["missing_seeds"] = "|".join(map(str, r["missing_seeds"]))
            w.writerow(out)


def main() -> None:
    args = parse_args()
    manifest_rows = load_manifest(args.manifest)
    points = build_points(manifest_rows, args.batch_size)
    if not points:
        print("No valid points found. Check manifest and output directories.")
        return

    agg = aggregate(points)
    print_table(agg)
    write_csv(agg, args.out_csv)
    print(f"\nSaved leaderboard CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
