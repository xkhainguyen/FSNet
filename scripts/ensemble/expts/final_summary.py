"""Final presentation-ready summary."""
import csv, os
DEF = '/orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/results.csv'
rows = list(csv.DictReader(open(DEF)))

def f(s, d=None):
    try: return float(s)
    except: return d

def find(method, **kw):
    """Find one matching row, batch=256, defaults."""
    for r in rows:
        if r['method'] != method: continue
        if int(r.get('batch_size','0')) != 256: continue
        ok = True
        for k, v in kw.items():
            if k == 'pK':
                if int(r.get('inference_perturb_k','0')) != v: ok = False
            elif k == 'peps':
                if abs(f(r.get('inference_perturb_eps'), 0) - v) > 1e-9: ok = False
            elif k == 'skip':
                if (r.get('skip_repair') in ('True','true', True)) != v: ok = False
            elif k == 'repIt':
                rmio = r.get('repair_max_iter_override') or ''
                rmio = int(rmio) if rmio not in ('','None') else None
                if rmio != v: ok = False
            elif k == 'M':
                if int(r.get('ensemble_size','1')) != v: ok = False
            elif k == 'post':
                if r.get('ensemble_post') != v: ok = False
            elif k == 'agg':
                if r.get('ensemble_agg') != v: ok = False
            elif k == 'run_contains':
                if v not in r['run']: ok = False
            elif k == 'run_excludes':
                if v in r['run']: ok = False
            if not ok: break
        if ok: return r
    return None

def m(r):
    if r is None: return ('-','-','-','-')
    o = f(r.get('objective')) ; mer = f(r.get('merit_mean')) ; iv = f(r.get('ineq_violation_l1_mean')) ; ev = f(r.get('eval_time_seconds'))
    return (
        f"{o:.3g}" if o else '-',
        f"{mer:.4g}" if mer else '-',
        f"{iv:.2e}" if iv is not None else '-',
        f"{ev:.1f}" if ev else '-',
    )

print("=" * 110)
print("FINAL SUMMARY — FSNet, SOCPProblem-100-50-50-10000, post + best_merit, batch=256")
print("=" * 110)
print(f"{'Config':<48} | {'Train s':>7} | {'Params':>7} | {'Obj':>7} | {'Merit':>10} | {'IneqVio':>10} | {'Eval s':>7}")
print("-" * 110)

baselines = [
    ('FSNet single (hdim=2048)',      'FSNet', dict(M=1, pK=0, skip=False, repIt=None, run_excludes='ens'), 527, '12.9M'),
    ('FSNet single (hdim=1024, member_0)', 'FSNet', dict(M=5, pK=0, skip=False, repIt=None, post='post', agg='best_merit', run_contains='ens5_post'), '~775', '3.3M'),
    ('FSNet vanilla ens5 (hdim=1024)', 'FSNet', dict(M=5, pK=0, skip=False, repIt=None, post='post', agg='best_merit', run_excludes='pert'), 3879, '16.5M'),
]

# Different approach: list rows with their training context
print("BASELINES:")
print(f"{'FSNet single (hdim=2048)':<48} | {'527':>7} | {'12.9M':>7}", end=' | ')
r = None
for row in rows:
    if row['method'] != 'FSNet': continue
    if int(row.get('batch_size','0')) != 256: continue
    if int(row.get('ensemble_size','1')) != 1: continue
    if int(row.get('inference_perturb_k','0')) > 0: continue
    if row.get('skip_repair') in ('True','true', True): continue
    if row.get('repair_max_iter_override') not in (None,'','None'): continue
    r = row; break
print(' | '.join(m(r)))

print(f"{'FSNet vanilla ens5 hdim=1024 post+best_merit':<48} | {'3879':>7} | {'16.5M':>7}", end=' | ')
for row in rows:
    if row['method'] != 'FSNet': continue
    if int(row.get('batch_size','0')) != 256: continue
    if int(row.get('ensemble_size','1')) != 5: continue
    if int(row.get('inference_perturb_k','0')) > 0: continue
    if row.get('skip_repair') in ('True','true', True): continue
    if row.get('repair_max_iter_override') not in (None,'','None'): continue
    if row.get('ensemble_post') != 'post': continue
    if row.get('ensemble_agg') != 'best_merit': continue
    # Pick the vanilla one (not the FGE one) — vanilla had training_time stored; FGE didn't.
    # We have 2 ens5 post+best_merit rows; the first is vanilla.
    r = row; break
print(' | '.join(m(r)))

print(f"{'FSNet FGE ens5 hdim=1024 (600 ep) post+best_merit':<48} | {'~5800':>7} | {'16.5M':>7}", end=' | ')
# Find FGE row — different timestamp
all_ens5_pbm = [row for row in rows
                if row['method']=='FSNet' and int(row.get('batch_size','0'))==256
                and int(row.get('ensemble_size','1'))==5
                and row.get('ensemble_post')=='post' and row.get('ensemble_agg')=='best_merit'
                and int(row.get('inference_perturb_k','0'))==0
                and not (row.get('skip_repair') in ('True','true', True))
                and row.get('repair_max_iter_override') in (None,'','None')]
# Two rows: vanilla (ts ~033538) and FGE (ts ~033752 in audit; FGE comes second by timestamp)
all_ens5_pbm.sort(key=lambda r: r['run'])
r = all_ens5_pbm[1] if len(all_ens5_pbm) >= 2 else None
print(' | '.join(m(r)))

print()
print("IDEA 2 — PERTURBATION (zero retraining cost on single FSNet member_0, hdim=1024):")
print(f"{'Config':<48} | {'Train s':>7} | {'Params':>7} | {'Obj':>7} | {'Merit':>10} | {'IneqVio':>10} | {'Eval s':>7}")
print("-" * 110)
for K in [5, 10, 20, 50, 100]:
    for eps in [0.05, 0.1]:
        r = find('FSNet', pK=K, peps=eps, run_contains='ens5_pre')
        if r is None: continue
        print(f"{'member_0 + K=%d eps=%s'%(K,eps):<48} | {'0':>7} | {'3.3M':>7}", end=' | ')
        print(' | '.join(m(r)))

print()
print("IDEA 2 — PERTURBATION (FSNet single hdim=2048):")
for K in [5, 10, 20, 50, 100]:
    for eps in [0.05, 0.1]:
        r = find('FSNet', pK=K, peps=eps, M=1, run_excludes='ens5')
        if r is None: continue
        print(f"{'single (hdim=2048) + K=%d eps=%s'%(K,eps):<48} | {'527':>7} | {'12.9M':>7}", end=' | ')
        print(' | '.join(m(r)))

print()
print("IDEA 3 — REPAIR ITERATION SWEEP (FSNet single hdim=2048):")
print(f"{'max_iter':<48} | {'Train s':>7} | {'Params':>7} | {'Eval s':>7} | {'Obj':>7} | {'Merit':>10} | {'IneqVio':>10}")
print("-" * 110)
for it in [None, 1, 5, 10, 20, 50]:
    if it is None:
        r = find('FSNet', M=1, skip=True, pK=0)
        label = '0 (skip)'
    else:
        r = find('FSNet', M=1, skip=False, repIt=it, pK=0)
        label = f'{it}'
    if r is None: continue
    print(f"{('max_iter=' + label):<48} | {'527':>7} | {'12.9M':>7}", end=' | ')
    print(' | '.join(m(r)))

print()
print("IDEA 1 — MHE (penalty, eq=10 ineq=10, hdim=1024×4, post+best_merit):")
print(f"{'Config':<48} | {'Train s':>7} | {'Params':>7} | {'Obj':>7} | {'Merit':>10} | {'IneqVio':>10} | {'Eval s':>7}")
print("-" * 110)
# Vanilla ens5 penalty
r = find('penalty', M=5, post='post', agg='best_merit', pK=0, skip=False, repIt=None, run_excludes='mhe')
print(f"{'penalty vanilla ens5 hdim=1024':<48} | {'1002':>7} | {'16.5M':>7}", end=' | ')
print(' | '.join(m(r)))
r = find('penalty', M=20, post='post', agg='best_merit', pK=0)
if r:
    print(f"{'penalty vanilla ens20 hdim=1024':<48} | {'~4000':>7} | {'66M':>7}", end=' | ')
    print(' | '.join(m(r)))
# MHE: find by run name pattern
for label, params, train_s, pattern in [
    ('penalty MHE M=5 s0 post+best_merit', '4.6M', '381', 'mhe5_seed0'),
    ('penalty MHE M=5 s1 post+best_merit', '4.6M', '380', 'mhe5_seed1'),
    ('penalty MHE M=10 s0 post+best_merit', '5.7M', '~700', 'mhe10_seed0'),
]:
    r = find('penalty', post='post', agg='best_merit', pK=0, run_contains=pattern)
    if r is None:
        r = find('penalty', post='pre', agg='best_merit', pK=0, run_contains=pattern)
    print(f"{label:<48} | {train_s:>7} | {params:>7}", end=' | ')
    print(' | '.join(m(r)))

