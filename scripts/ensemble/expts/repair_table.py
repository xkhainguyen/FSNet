"""Table of repair iter vs Merit/IneqViol/EqViol for FSNet single."""
import csv, os
DEF = '/orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/results.csv'
rows = list(csv.DictReader(open(DEF)))

def f(s, d=None):
    try: return float(s)
    except: return d

print(f"\n=== FSNet single (hdim=2048) — repair-iter sweep ===")
print(f"{'max_iter':>10} | {'skip':>4} | {'Obj':>7} | {'Merit':>10} | {'IneqVio':>10} | {'EqVio':>10}")
print("-"*70)
records = []
for r in rows:
    if r['method'] != 'FSNet': continue
    if int(r.get('batch_size','0')) != 256: continue
    if int(r.get('ensemble_size','1')) > 1: continue
    if int(r.get('inference_perturb_k','0')) > 1: continue
    skip = r.get('skip_repair') in ('True','true', True)
    rep_it = r.get('repair_max_iter_override') or ''
    rep_it = int(rep_it) if rep_it not in ('', 'None') else None
    if skip or rep_it is not None:
        records.append((rep_it if rep_it is not None else 0, skip, r))
records.sort(key=lambda x: (x[0], -int(x[1])))
for max_it, skip, r in records:
    print(f"{str(max_it) if not skip else '0':>10} | {'Y' if skip else '-':>4} | {f(r['objective']):>7.3g} | {f(r['merit_mean']):>10.4g} | {f(r['ineq_violation_l1_mean']):>10.4g} | {f(r['eq_violation_l1_mean']):>10.4g}")

# Also baseline (default = max_iter=50)
print("\n(default max_iter=50 baseline shown above; Merit 85.05 = full L-BFGS)")
