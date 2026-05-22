"""Table of perturbation results on member_0 (apples-to-apples hdim=1024)."""
import csv, os
DEF = '/orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/results.csv'
rows = list(csv.DictReader(open(DEF)))

# The member_0 evals saved with tag "ens5_pre_..._pertK..." or "ens5_pre_..._norepair_..."
# because they inherit the ensemble_size=5 from the saved config.

def f(s, d=None):
    try: return float(s)
    except: return d

def fmt(x, p=4):
    if x is None: return '-'
    if x == 0: return '0'
    a = abs(x)
    if a >= 1e4 or a < 1e-2: return f"{x:.{p-1}g}"
    return f"{x:.{p}g}"

print("\n=== FSNet member_0 of ens5_vanilla (hdim=1024) — perturbation sweep ===")
print(f"{'K':>4} | {'eps':>6} | {'Obj':>7} | {'Merit':>10} | {'OptGap%':>8} | {'IneqVio':>10}")
print("-"*60)

# Find member_0 rows: ens5_pre_mean_pertK*_eps*  (inherits ens=5 from config)
records = {}
for r in rows:
    if r['method'] != 'FSNet': continue
    if int(r.get('batch_size','0')) != 256: continue
    K = int(r.get('inference_perturb_k','0'))
    if K < 1: continue
    eps = f(r.get('inference_perturb_eps'), 0)
    # Distinguish member_0 (hdim=1024, train_time ~775 or so, params ~3.3M) from real single (hdim=2048, params ~12.9M)
    run = r['run']
    if 'ens5_pre_' in run or 'ens5_pre_mean' in run:
        # member_0 (configed as ensemble but it's a single member)
        records[(K, eps, 'mem0')] = r
    elif r.get('inference_perturb_k','0') != '0':
        records[(K, eps, 'single')] = r

# Single (hdim=2048) baseline + perturb
print("\nSingle FSNet (hdim=2048):")
for K in [0, 5, 10, 20, 50, 100]:
    for eps in [0.01, 0.05, 0.1, 0.2]:
        if K == 0 and eps != 0.01: continue  # baseline only once
        key = (K if K > 0 else 0, eps if K > 0 else 0, 'single')
        if K == 0:
            # baseline
            for r in rows:
                if r['method'] != 'FSNet': continue
                if int(r.get('batch_size','0')) != 256: continue
                if int(r.get('ensemble_size','1')) != 1: continue
                if int(r.get('inference_perturb_k','0')) != 0: continue
                if r.get('skip_repair') in ('True','true', True): continue
                if r.get('repair_max_iter_override') not in (None,'','None'): continue
                obj = fmt(f(r.get('objective')))
                mer = fmt(f(r.get('merit_mean')))
                ogm = fmt(f(r.get('opt_gap_mean')))
                iv = fmt(f(r.get('ineq_violation_l1_mean')))
                print(f"{'1':>4} | {'-':>6} | {obj:>7} | {mer:>10} | {ogm:>8} | {iv:>10}")
                break
            break
        r = records.get(key)
        if r:
            obj = fmt(f(r.get('objective')))
            mer = fmt(f(r.get('merit_mean')))
            ogm = fmt(f(r.get('opt_gap_mean')))
            iv = fmt(f(r.get('ineq_violation_l1_mean')))
            print(f"{K:>4} | {eps:>6} | {obj:>7} | {mer:>10} | {ogm:>8} | {iv:>10}")

print("\nMember_0 (hdim=1024, apples-to-apples with vanilla ens5):")
for K in [5, 10, 20, 50, 100]:
    for eps in [0.01, 0.05, 0.1, 0.2]:
        r = records.get((K, eps, 'mem0'))
        if r:
            obj = fmt(f(r.get('objective')))
            mer = fmt(f(r.get('merit_mean')))
            ogm = fmt(f(r.get('opt_gap_mean')))
            iv = fmt(f(r.get('ineq_violation_l1_mean')))
            print(f"{K:>4} | {eps:>6} | {obj:>7} | {mer:>10} | {ogm:>8} | {iv:>10}")
