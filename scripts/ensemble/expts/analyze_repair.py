"""Analyze repair-ablation results: max_iter / skip → Merit."""
import csv, os
DEF = '/orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/results.csv'
rows = list(csv.DictReader(open(DEF)))

def f(s, d=None):
    try: return float(s)
    except: return d

groups = {}
for r in rows:
    if int(r.get('batch_size','0')) != 256: continue
    skip = r.get('skip_repair') in ('True','true', True)
    rep_it = r.get('repair_max_iter_override') or ''
    rep_it = int(rep_it) if rep_it not in ('', 'None') else None
    if skip is False and rep_it is None: continue  # baseline
    # Skip ensemble configs for this view
    if int(r.get('ensemble_size','1')) > 1: continue
    if int(r.get('inference_perturb_k','0')) > 1: continue
    key = (r['method'], skip, rep_it)
    groups[key] = r

print(f"{'method':<8} | {'skip':>4} | {'max_iter':>8} | {'Obj':>7} | {'Merit':>10} | {'IneqVio':>10} | {'EqVio':>10}")
print("-"*70)
for key, r in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or 0)):
    m, skip, rep_it = key
    obj = f(r.get('objective'))
    mer = f(r.get('merit_mean'))
    iv = f(r.get('ineq_violation_l1_mean'))
    ev = f(r.get('eq_violation_l1_mean'))
    print(f"{m:<8} | {'Y' if skip else '-':>4} | {str(rep_it or '-'):>8} | {obj:>7.3g} | {mer:>10.4g} | {iv:>10.4g} | {ev:>10.4g}")
