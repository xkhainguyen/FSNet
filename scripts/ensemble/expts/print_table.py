"""Quick pretty-print of the aggregated results CSV.

Usage: python scripts/ensemble/expts/print_table.py [--csv path]
"""
import csv, argparse, sys, os

DEF = '/orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/results.csv'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=DEF)
    p.add_argument('--method', default=None)
    p.add_argument('--bs', type=int, default=256)
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        print(f"missing {args.csv}; run aggregate_results.py first")
        return

    rows = list(csv.DictReader(open(args.csv)))
    rows = [r for r in rows if int(r.get('batch_size','0')) == args.bs]
    if args.method:
        rows = [r for r in rows if r['method'] == args.method]
    rows.sort(key=lambda r: (r['method'], r['run']))

    cols = [
        ('run', 60),
        ('M', 4),
        ('post', 5),
        ('agg', 11),
        ('skip', 5),
        ('repIt', 6),
        ('pertK', 6),
        ('eps', 5),
        ('Obj', 8),
        ('Merit', 12),
        ('OptGap%', 8),
        ('IneqVio', 10),
        ('EqVio', 9),
        ('tr_s', 7),
        ('ev_s', 6),
    ]
    print(' | '.join(f"{c.ljust(w)}" for c,w in cols))
    print('-+-'.join('-'*w for c,w in cols))
    for r in rows:
        run = r['run']
        # Trim run name
        run_short = run.replace('20260522-', '').replace('_seed', '_s')[:cols[0][1]]
        vals = [
            run_short,
            r.get('ensemble_size','1'),
            (r.get('ensemble_post','-') or '-')[:4],
            (r.get('ensemble_agg','-') or '-')[:11],
            'Y' if r.get('skip_repair') in ('True','true', True) else '-',
            r.get('repair_max_iter_override','-') or '-',
            r.get('inference_perturb_k','0') or '0',
            r.get('inference_perturb_eps','0') or '0',
            f"{float(r['objective']):.3g}" if r.get('objective') not in (None,'') else '-',
            f"{float(r['merit_mean']):.4g}" if r.get('merit_mean') not in (None,'') else '-',
            f"{float(r['opt_gap_mean']):.4g}" if r.get('opt_gap_mean') not in (None,'') else '-',
            f"{float(r['ineq_violation_l1_mean']):.4g}" if r.get('ineq_violation_l1_mean') not in (None,'') else '-',
            f"{float(r['eq_violation_l1_mean']):.4g}" if r.get('eq_violation_l1_mean') not in (None,'') else '-',
            f"{float(r['training_time_seconds']):.0f}" if r.get('training_time_seconds') not in (None,'') else '-',
            f"{float(r['eval_time_seconds']):.1f}" if r.get('eval_time_seconds') not in (None,'') else '-',
        ]
        print(' | '.join(str(v).ljust(w)[:w] for v, (_,w) in zip(vals, cols)))

if __name__ == '__main__':
    main()
