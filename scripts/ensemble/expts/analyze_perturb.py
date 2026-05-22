"""Analyze perturbation eval results: K, eps -> Merit / Obj / OptGap."""
import csv, os
DEF = '/orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/results.csv'

if not os.path.isfile(DEF):
    raise SystemExit("aggregate first")

rows = list(csv.DictReader(open(DEF)))

def f(s, d=None):
    try: return float(s)
    except: return d

# Group: method × pertK × eps × dist
groups = {}
for r in rows:
    if int(r.get('batch_size','0')) != 256: continue
    K = int(r.get('inference_perturb_k','0'))
    if K < 2: continue  # baseline (single eval) only
    key = (r['method'], K, f(r['inference_perturb_eps'], 0), r.get('inference_perturb_dist','gauss'))
    groups[key] = r

print(f"{'method':<8} | {'K':>3} | {'eps':>6} | {'dist':<10} | {'Obj':>7} | {'Merit':>10} | {'IneqVio':>10}")
print("-"*70)
for key, r in sorted(groups.items()):
    m, K, eps, dist = key
    obj = f(r.get('objective'))
    mer = f(r.get('merit_mean'))
    iv = f(r.get('ineq_violation_l1_mean'))
    print(f"{m:<8} | {K:>3} | {eps:>6.3g} | {dist:<10} | {obj:>7.3g} | {mer:>10.4g} | {iv:>10.4g}")
