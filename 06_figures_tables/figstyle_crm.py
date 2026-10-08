# -*- coding: utf-8 -*-
"""
figstyle_crm.py — BilinGuard figure style module, modelled on the design language of
Gao et al., "Multi-modal AI-enabled steatotic liver disease diagnostics using facial
images and metabolomics", Cell Reports Medicine 2026 (the "3D-FAICE" paper).

Design rules implemented here (see FIGURE_TABLE_REDESIGN_CRM.md):
  * internal cohort = blue (#2B6CB0), external cohort = amber (#D97706)
  * positive / negative = red #C0392B / deep blue #2E5FA3
  * panel labels = single bold capital letter at the axes' top-left corner
    (NO in-figure suptitle, NO descriptive panel titles; explanation lives in the caption)
  * no gridlines on scatter/ROC panels; outward ticks; Arial 7pt base
  * 95% CI bands alpha=0.15; bar charts carry CI whiskers and */**/***/ significance
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

# ----------------------------------------------------------------------------- palette
INT   = '#2B6CB0'   # internal cohort — blue
EXT   = '#D97706'   # external cohort — amber
POS   = '#C0392B'   # positive / significant / BilinGuard star
NEG   = '#2E5FA3'   # negative
FUS   = '#2E8B6E'   # fusion
GREY  = '#6B7280'
LGREY = '#9CA3AF'
READER = '#C0392B'  # clinician points
MODEL_STAR = '#DC2626'

ARCH_COL = {'ConvNeXt': '#4C78A8', 'ViT': '#F58518', 'Swin': '#54A24B',
            'EfficientNet': '#B279A2', 'Ensemble': '#1A1A1A'}
ARCH_LS  = {'ConvNeXt': '-', 'ViT': '--', 'Swin': '-.', 'EfficientNet': ':', 'Ensemble': '-'}
SCOPE_COL = {'Face': INT, 'Eyelid': EXT, 'Fusion': FUS}
TASK_COL = {'Binary Screening': '#2B6CB0', 'Ternary Grading': '#7C9A6D', 'DBIL': '#6A5F9E',
            'IBIL': '#2E8B6E', 'Jaundice Type': '#C0392B', 'Child-Pugh': '#D97706', 'MELD': '#8B2E2E'}
TASK_SHORT = {'Binary Screening': 'Binary', 'Ternary Grading': 'TBIL', 'DBIL': 'DBIL',
              'IBIL': 'IBIL', 'Jaundice Type': 'Type', 'Child-Pugh': 'CP', 'MELD': 'MELD'}

FPR_GRID = np.linspace(0, 1, 101)
SEED = 42


def set_rc():
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 7.5,
        'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
        'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'legend.frameon': False, 'legend.fontsize': 6.5,
        'figure.dpi': 600, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})


def panel_letter(ax, letter):
    """Cell-Press style: one bold capital at the top-left of the panel."""
    ax.set_title(f'{letter}', loc='left', fontsize=11, fontweight='bold', pad=3)


def roc_ax(ax):
    ax.plot([0, 1], [0, 1], ':', color=LGREY, lw=0.7)
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.grid(False)


def sig(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))


def save(fig, path_base):
    """Robust save: write to a temp file, validate (PNG via PIL, SVG via XML +
    NUL check), copy into place, re-verify, retry. This machine occasionally
    zero-fills or truncates large writes, so every save self-verifies."""
    import os, shutil, time, tempfile
    import xml.etree.ElementTree as ET
    from PIL import Image as _Img
    tmpdir = tempfile.gettempdir()
    for ext in ('png', 'svg', 'pdf'):
        final = path_base + '.' + ext
        done = False
        for k in range(5):
            tmp = os.path.join(tmpdir, 'figsave_%d.%s' % (os.getpid(), ext))
            try:
                fig.savefig(tmp)
            except Exception:
                time.sleep(0.4); continue
            ok = True
            try:
                if ext == 'png':
                    _Img.open(tmp).verify()
                elif ext == 'pdf':
                    ok = os.path.getsize(tmp) > 2000
                else:
                    raw = open(tmp, 'rb').read()
                    if raw.count(b'\x00'):
                        ok = False
                    else:
                        ET.fromstring(raw)
            except Exception:
                ok = False
            if ok:
                try:
                    shutil.copy2(tmp, final)
                    if ext == 'png':
                        _Img.open(final).verify()
                        ok = True
                    elif ext == 'pdf':
                        ok = os.path.getsize(final) > 2000
                    else:
                        ok = (open(final, 'rb').read() == open(tmp, 'rb').read())
                except Exception:
                    ok = False
            if ok:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                done = True
                break
            time.sleep(0.4)
        if not done:
            print('WARNING: could not verify %s after 5 attempts' % final)
        print('saved', final)
    plt.close(fig)


# ------------------------------------------------------------- data / stats helpers
def pd_norm(d):
    return {pid: (np.array(v[0], dtype=float), int(v[1])) for pid, v in d.items()}


def merge_ens(dicts):
    dicts = [d for d in dicts if d]
    common = sorted(set.intersection(*[set(d.keys()) for d in dicts]))
    return {pid: (np.mean([d[pid][0] for d in dicts], 0), dicts[0][pid][1]) for pid in common}


def get_tP(pd_):
    pids = sorted(pd_.keys())
    t = np.array([pd_[p][1] for p in pids])
    P = np.array([pd_[p][0] for p in pids])
    return t, P


def roc_band(pd_, nc=2, n_boot=300):
    """Return fpr, tpr, auc(point), lo, hi (bootstrap band) for a prob-dict."""
    t, P = get_tP(pd_)
    if nc == 2:
        fpr, tpr, _ = roc_curve(t, P[:, 1])
        a = auc(fpr, tpr)
        rng = np.random.RandomState(SEED); tprs = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(t), len(t))
            if len(np.unique(t[idx])) < 2: continue
            f, tt, _ = roc_curve(t[idx], P[idx][:, 1])
            tprs.append(np.interp(FPR_GRID, f, tt))
        lo = np.percentile(tprs, 2.5, axis=0) if tprs else None
        hi = np.percentile(tprs, 97.5, axis=0) if tprs else None
        return fpr, tpr, a, lo, hi
    yb = label_binarize(t, classes=list(range(nc)))
    fpr, tpr, _ = roc_curve(yb.ravel(), P.ravel())
    try:
        a = roc_auc_score(t, P, multi_class='ovr', labels=list(range(nc)))
    except Exception:
        a = float('nan')
    rng = np.random.RandomState(SEED); tprs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(t), len(t))
        if len(np.unique(t[idx])) < nc: continue
        yb2 = label_binarize(t[idx], classes=list(range(nc)))
        f, tt, _ = roc_curve(yb2.ravel(), P[idx].ravel())
        tprs.append(np.interp(FPR_GRID, f, tt))
    lo = np.percentile(tprs, 2.5, axis=0) if tprs else None
    hi = np.percentile(tprs, 97.5, axis=0) if tprs else None
    return fpr, tpr, a, lo, hi


def roc_band_arrays(t, prob1d, n_boot=300):
    """Binary ROC band from raw arrays (external frozen etc.)."""
    t = np.asarray(t).astype(int); p = np.asarray(prob1d).astype(float)
    fpr, tpr, _ = roc_curve(t, p); a = auc(fpr, tpr)
    rng = np.random.RandomState(SEED); tprs, aucs = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, len(t), len(t))
        if len(np.unique(t[idx])) < 2: continue
        f, tt, _ = roc_curve(t[idx], p[idx])
        tprs.append(np.interp(FPR_GRID, f, tt)); aucs.append(auc(f, tt))
    return fpr, tpr, a, np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0), \
        np.percentile(aucs, [2.5, 97.5])


def macro_ovr_roc_arrays(t, P, nc=3, n_boot=300):
    """Macro OvR ROC band from label array + prob matrix."""
    t = np.asarray(t); P = np.asarray(P)
    yb = label_binarize(t, classes=list(range(nc)))
    mean_tpr = np.zeros_like(FPR_GRID)
    for c in range(nc):
        f, tt, _ = roc_curve(yb[:, c], P[:, c])
        mean_tpr += np.interp(FPR_GRID, f, tt)
    mean_tpr /= nc
    a = roc_auc_score(t, P, multi_class='ovr', labels=list(range(nc)))
    rng = np.random.RandomState(SEED); tprs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(t), len(t))
        ok = True; mb = np.zeros_like(FPR_GRID)
        for c in range(nc):
            if len(np.unique(yb[idx, c])) < 2: ok = False; break
            f, tt, _ = roc_curve(yb[idx, c], P[idx, c])
            mb += np.interp(FPR_GRID, f, tt)
        if not ok: continue
        tprs.append(mb / nc)
    lo = np.percentile(tprs, 2.5, axis=0) if tprs else None
    hi = np.percentile(tprs, 97.5, axis=0) if tprs else None
    return FPR_GRID, mean_tpr, a, lo, hi


def plot_model_rocs(ax, prob_dict, nc, archs=(('convnext', 'ConvNeXt'), ('vit', 'ViT'),
                    ('swin', 'Swin'), ('efficientnet', 'EfficientNet'))):
    """4 backbone thin curves + thick ensemble with CI band (legend AUCs)."""
    dicts = []
    for key, name in archs:
        if key not in prob_dict or not prob_dict[key]: continue
        pd_ = pd_norm(prob_dict[key]); dicts.append(pd_)
        fpr, tpr, a, _, _ = roc_band(pd_, nc, n_boot=0)
        ax.plot(fpr, tpr, ARCH_LS[name], color=ARCH_COL[name], lw=0.9,
                label=f'{name} ({a:.3f})')
    ens = merge_ens(dicts)
    fpr, tpr, a, lo, hi = roc_band(ens, nc)
    ax.plot(fpr, tpr, '-', color=ARCH_COL['Ensemble'], lw=1.6, label=f'Ensemble ({a:.3f})')
    if lo is not None:
        ax.fill_between(FPR_GRID, lo, hi, color=ARCH_COL['Ensemble'], alpha=0.15, lw=0)
    roc_ax(ax)
    ax.legend(loc='lower right', fontsize=5.5, handlelength=1.6)
    return a
