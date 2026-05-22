"""Aggregate eval test_summary.yaml files into a CSV table.

Walks results/ recursively for eval-tagged directories (containing 'eval_' in
the name), parses test_summary.yaml, emits a tidy CSV.

Usage:
    python scripts/ensemble/expts/aggregate_results.py [--since YYYYMMDD-HHMMSS]
                                                       [--out results.csv]
"""

import os
import re
import sys
import yaml
import argparse
import csv
from datetime import datetime

ROOT = '/orcd/scratch/orcd/008/khain/FSNet/results'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--since', default='20260522-000000',
                   help='Only include eval dirs whose timestamp prefix >= this.')
    p.add_argument('--out', default='/orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/results.csv')
    p.add_argument('--method', default=None, help='Filter to a specific method (e.g. penalty, FSNet)')
    p.add_argument('--include_train', action='store_true',
                   help='Also include the original test_summary.yaml from training runs.')
    return p.parse_args()


def find_test_summaries(root, since=None, include_train=False):
    """Yield (dir_path, test_summary_yaml_path) for eval result dirs."""
    pat = re.compile(r'^(\d{8}-\d{6})_(.+)$')
    for prob_type in os.listdir(root):
        d1 = os.path.join(root, prob_type)
        if not os.path.isdir(d1): continue
        for prob_name in os.listdir(d1):
            d2 = os.path.join(d1, prob_name)
            if not os.path.isdir(d2): continue
            for prob_inst in os.listdir(d2):
                d3 = os.path.join(d2, prob_inst)
                if not os.path.isdir(d3): continue
                for run in os.listdir(d3):
                    if since and run < since:
                        continue
                    m = pat.match(run)
                    if not m: continue
                    is_eval = '_eval_' in run
                    if not is_eval and not include_train:
                        continue
                    run_dir = os.path.join(d3, run)
                    ts_path = os.path.join(run_dir, 'test_summary.yaml')
                    if os.path.isfile(ts_path):
                        yield run_dir, ts_path, prob_type, prob_name, prob_inst


def parse_one(run_dir, ts_path, prob_type, prob_name, prob_inst):
    with open(ts_path) as f:
        s = yaml.safe_load(f)
    run = os.path.basename(run_dir)

    rec = {
        'run': run,
        'prob_type': prob_type,
        'prob_name': prob_name,
        'prob_inst': prob_inst,
        'method': s.get('method', ''),
        'seed': s.get('seed', ''),
        'ensemble_size': s.get('ensemble_size', 1),
        'ensemble_post': s.get('ensemble_post', 'pre'),
        'ensemble_agg': s.get('ensemble_agg', 'mean'),
        'skip_repair': s.get('skip_repair', False),
        'repair_max_iter_override': s.get('repair_max_iter_override', None),
        'inference_perturb_k': s.get('inference_perturb_k', 0),
        'inference_perturb_eps': s.get('inference_perturb_eps', 0.0),
        'inference_perturb_dist': s.get('inference_perturb_dist', 'gauss'),
        'training_time_seconds': s.get('training_time_seconds', None),
        'eval_time_seconds': s.get('eval_time_seconds', None),
        'is_eval': '_eval_' in run,
    }

    bs_results = s.get('test', {})
    if not bs_results:
        # training run format
        bsc = s.get('batch_size_comparison', {})
        for bs, r in bsc.items():
            if 'error' in r: continue
            rec_bs = dict(rec, batch_size=int(bs))
            rec_bs.update({k: r['metrics'].get(k) for k in
                           ['objective','opt_gap_mean','opt_gap_std',
                            'eq_violation_l1_mean','eq_violation_l1_max',
                            'ineq_violation_l1_mean','ineq_violation_l1_max',
                            'merit_mean','solution_distance_mean']})
            yield rec_bs
        return

    for bs, m in bs_results.items():
        if 'error' in m: continue
        rec_bs = dict(rec, batch_size=int(bs))
        rec_bs.update({k: m.get(k) for k in
                       ['objective','opt_gap_mean','opt_gap_std',
                        'eq_violation_l1_mean','eq_violation_l1_max',
                        'ineq_violation_l1_mean','ineq_violation_l1_max',
                        'merit_mean','solution_distance_mean']})
        yield rec_bs


def main():
    args = parse_args()
    rows = []
    for entry in find_test_summaries(ROOT, since=args.since, include_train=args.include_train):
        run_dir, ts_path, prob_type, prob_name, prob_inst = entry
        try:
            for rec in parse_one(run_dir, ts_path, prob_type, prob_name, prob_inst):
                if args.method and rec['method'] != args.method:
                    continue
                rows.append(rec)
        except Exception as e:
            print(f"WARN: failed to parse {ts_path}: {e}", file=sys.stderr)

    if not rows:
        print("No rows.")
        return

    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == '__main__':
    main()
