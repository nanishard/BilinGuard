# -*- coding: utf-8 -*-
"""
PAIRED fusion comparison on shared held-out patients (leakage-free).

Data sources per task:
  Binary/Ternary: face = canonical V2 probs (held out from face training);
                  eyelid = fusion-dedicated models (fusion_shared_probs, held out)
  DBIL/IBIL:      face & eyelid from canonical V2 (same manifest split; intersection)
  Type:           face & eyelid from typev2_probs (same rebuilt split, n=44)
  CP/MELD:        face & eyelid from canonical V2 (face val ∩ eyelid val)

For each task, on the SAME shared patients:
  face best AUC, eyelid best AUC, best single-view AUC,
  fusion AUC (0.5/0.5 per-arch then best), paired bootstrap ΔAUC (fusion − best single) with 95% CI.
Output: results/fusion_paired_comparison.json
"""
import os, json, numpy as np
from sklearn.metrics import roc_auc_score

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
SEED = 42
V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))
FS = json.load(open(os.path.join(RES, 'fusion_shared_probs.json'), encoding='utf-8'))
T2 = json.load(open(os.path.join(RES, 'typev2_probs.json'), encoding='utf-8'))
ARCHS = ['convnext', 'vit', 'swin', 'efficientnet']


def pd_norm(d):
    return {pid: (np.array(v[0], dtype=float), int(v[1])) for pid, v in d.items()}


def subset(d, pids):
    return {p: d[p] for p in pids if p in d}


def auc_val(t, P, nc):
    if len(np.unique(t)) < 2: return float('nan')
    try:
        return roc_auc_score(t, P[:, 1]) if nc == 2 else \
            roc_auc_score(t, P, multi_class='ovr', labels=list(range(nc)))
    except Exception:
        return float('nan')


def best_of(prob_dicts, pids, nc):
    """Return (auc, arch, t, P) for the best arch on pids."""
    best = (-1, None, None, None)
    for arch, d in prob_dicts.items():
        sub = subset(d, pids)
        if not sub: continue
        t = np.array([v[1] for v in sub.values()])
        P = np.array([v[0] for v in sub.values()])
        a = auc_val(t, P, nc)
        if not np.isnan(a) and a > best[0]:
            best = (a, arch, t, P)
    return best


def fuse(face_d, eye_d, pids):
    out = {}
    for a in ARCHS:
        if a not in face_d or a not in eye_d: continue
        fa, ea = face_d[a], eye_d[a]
        out[a] = {p: ((fa[p][0] + ea[p][0]) / 2, fa[p][1])
                  for p in pids if p in fa and p in ea}
    return out


def paired_delta(t, P_fus, P_best, nc, n_boot=1000):
    rng = np.random.RandomState(SEED)
    n = len(t)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(t[idx])) < 2: continue
        af = auc_val(t[idx], P_fus[idx], nc)
        ab = auc_val(t[idx], P_best[idx], nc)
        if not (np.isnan(af) or np.isnan(ab)):
            diffs.append(af - ab)
    if not diffs: return float('nan'), float('nan'), float('nan')
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def analyze(task, nc, face_src, eye_src, shared_pids):
    face_d = {a: pd_norm(face_src[a]) for a in ARCHS if a in face_src}
    eye_d = {a: pd_norm(eye_src[a]) for a in ARCHS if a in eye_src}
    fus_d = fuse(face_d, eye_d, shared_pids)
    fa, fa_arch, _, _ = best_of(face_d, shared_pids, nc)
    ea, ea_arch, _, _ = best_of(eye_d, shared_pids, nc)
    fu, fu_arch, tf, Pf = best_of(fus_d, shared_pids, nc)
    if fa >= ea:
        bs_a, bs_scope, tb, Pb = fa, 'face', *best_of(face_d, shared_pids, nc)[2:]
    else:
        bs_a, bs_scope, tb, Pb = ea, 'eyelid', *best_of(eye_d, shared_pids, nc)[2:]
    dm, dl, dh = paired_delta(tf, Pf, Pb, nc)
    return dict(n=len(shared_pids),
                face=dict(auc=round(fa, 4), arch=fa_arch),
                eyelid=dict(auc=round(ea, 4), arch=ea_arch),
                best_single=dict(auc=round(bs_a, 4), scope=bs_scope),
                fusion=dict(auc=round(fu, 4), arch=fu_arch),
                delta=dict(mean=round(dm, 4), lo=round(dl, 4), hi=round(dh, 4)))


results = {}
results['Binary Screening'] = analyze(
    'Binary Screening', 2,
    V2['Binary Screening']['face'], FS['Binary Screening']['eyelid'],
    FS['Binary Screening']['val_pids'])
results['Ternary Grading'] = analyze(
    'Ternary Grading', 3,
    V2['Ternary Grading']['face'], FS['Ternary Grading']['eyelid'],
    FS['Ternary Grading']['val_pids'])
for task in ['DBIL', 'IBIL']:
    face_src, eye_src = V2[task]['face'], V2[task]['eyelid']
    shared = sorted(set.intersection(*[set(face_src[a].keys()) for a in ARCHS if a in face_src],
                                     *[set(eye_src[a].keys()) for a in ARCHS if a in eye_src]))
    results[task] = analyze(task, 3, face_src, eye_src, shared)
shared_type = sorted(set.intersection(*[set(T2['face'][a].keys()) for a in ARCHS if a in T2['face']],
                                      *[set(T2['eyelid'][a].keys()) for a in ARCHS if a in T2['eyelid']]))
results['Jaundice Type'] = analyze('Jaundice Type', 2, T2['face'], T2['eyelid'], shared_type)
for task in ['Child-Pugh', 'MELD']:
    face_src, eye_src = V2[task]['face'], V2[task]['eyelid']
    shared = sorted(set.intersection(*[set(face_src[a].keys()) for a in ARCHS if a in face_src],
                                     *[set(eye_src[a].keys()) for a in ARCHS if a in eye_src]))
    results[task] = analyze(task, 3, face_src, eye_src, shared)

out = os.path.join(RES, 'fusion_paired_comparison.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"{'Task':<18}{'n':>4} {'Face':>8} {'Eyelid':>8} {'BestSingle':>16} {'Fusion':>8} {'dAUC[95%CI]':>24}")
print('-' * 95)
for task, r in results.items():
    bs = f"{r['best_single']['auc']:.3f}({r['best_single']['scope']})"
    print(f"{task:<18}{r['n']:>4} {r['face']['auc']:>8.3f} {r['eyelid']['auc']:>8.3f} {bs:>16} "
          f"{r['fusion']['auc']:>8.3f} {r['delta']['mean']:+.3f} [{r['delta']['lo']:+.3f},{r['delta']['hi']:+.3f}]")
print(f'\nSaved: {out}')
