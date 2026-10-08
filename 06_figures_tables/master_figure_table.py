# -*- coding: utf-8 -*-
"""
MASTER SCRIPT: Bootstrap CI for new binary models + regenerate ALL figures
(English-only text to fix CJK garbled characters) + tables + web update.

Step 1: Bootstrap CI for 4 new face binary models
Step 2: Regenerate Figure 2/3/4 + Supplementary (all English text)
Step 3: Update tables
"""
import os, json, random, re, numpy as np, pandas as pd, cv2
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torchvision import transforms
import timm
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, f1_score,
                              confusion_matrix, precision_recall_curve, average_precision_score,
                              brier_score_loss)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'tables')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
FIG = os.path.join(RES, 'figures', 'manuscript')
SUPP = os.path.join(RES, 'figures', 'supplementary')
os.makedirs(FIG, exist_ok=True); os.makedirs(SUPP, exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; IMG_SIZE = 224; N_BOOT = 1000
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# Use a font that supports both English and Chinese
plt.rcParams.update({'font.sans-serif': ['Arial', 'DejaVu Sans', 'SimHei'],
                      'font.family': 'sans-serif', 'font.size': 7, 'axes.linewidth': 0.5,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})

C = {'convnext': '#264653', 'vit': '#2A9D8F', 'effnet': '#E9C46A',
     'swin': '#E76F51', 'yolo': '#6A4C93', 'doctor': '#457B9D',
     'model': '#1D3557', 'mild': '#F4D03F', 'moderate': '#E67E22', 'severe': '#E74C3C'}

ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def read_img(p):
    try:
        fb = np.fromfile(p, dtype=np.uint8); img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except: return None

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab); l = cv2.createCLAHE(3.0,(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)

def predict(model, paths, n=12):
    valid = [p for p in paths[:n] if os.path.exists(p)]
    if not valid: return None
    ts = []
    for p in valid:
        img = read_img(p)
        if img is None: continue
        ts.append(ev_tf(Image.fromarray(clahe(img)).convert('RGB')).unsqueeze(0))
    if not ts: return None
    with torch.no_grad(): return F.softmax(model(torch.cat(ts).to(DEV)),1).mean(0).cpu().numpy()

def load_m(bb, ckpt, nc):
    p = os.path.join(MODEL, ckpt)
    if not os.path.exists(p): return None
    m = timm.create_model(bb, pretrained=False, num_classes=nc).to(DEV)
    try: m.load_state_dict(torch.load(p, map_location=DEV, weights_only=True))
    except: return None
    m.eval(); return m

def collect():
    pats = {}
    for cat in ['normal','mild','moderate','severe']:
        cd = os.path.join(DATA, cat)
        if not os.path.exists(cd): continue
        for pid in os.listdir(cd):
            p = os.path.join(cd, pid)
            if not os.path.isdir(p): continue
            faces = sorted([os.path.join(p,f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
            if faces: pats[pid] = {'faces': faces, 'cat': cat}
    mdf = pd.read_csv(MANIFEST)
    for _,r in mdf.iterrows():
        if r['patient_id'] in pats and r['n_eyelid']>0:
            pats[r['patient_id']]['eyelids'] = json.loads(r['eyelid_images'])
    return pats

def split(pid_grade, ratio=0.2):
    bg = {}
    for pid,g in pid_grade.items(): bg.setdefault(g,[]).append(pid)
    tr,va = set(),set()
    for g,pids in bg.items():
        random.seed(SEED); random.shuffle(pids)
        n = max(1,int(len(pids)*ratio))
        va.update(pids[:n]); tr.update(pids[n:])
    return tr,va

def compute_metrics(true, probs, nc=2):
    pred = probs.argmax(1); cm = confusion_matrix(true, pred, labels=list(range(nc)))
    try: auc = roc_auc_score(true, probs[:,1]) if nc==2 else roc_auc_score(true, probs, multi_class='ovr', labels=list(range(nc)))
    except: auc = 0
    try: ap = average_precision_score(true, probs[:,1]) if nc==2 else np.mean([average_precision_score((true==c).astype(int), probs[:,c]) for c in range(nc)])
    except: ap = 0
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro', labels=list(range(nc)), zero_division=0)
    if nc==2:
        sens = cm[1,1]/max(cm[1,:].sum(),1); spec = cm[0,0]/max(cm[0,:].sum(),1)
    else:
        sl,sp=[],[]
        for c in range(nc):
            tp=cm[c,c];fn=cm[c,:].sum()-tp;fp=cm[:,c].sum()-tp;tn=cm.sum()-tp-fn-fp
            sl.append(tp/max(tp+fn,1)); sp.append(tn/max(tn+fp,1))
        sens=float(np.mean(sl)); spec=float(np.mean(sp))
    return {'auc':float(auc),'ap':float(ap),'acc':float(acc),'f1':float(f1),'sens':float(sens),'spec':float(spec)}

def boot_ci(true, probs, nc=2, n_boot=N_BOOT):
    rng = np.random.RandomState(SEED); n=len(true)
    res = {k:[] for k in ['auc','ap','acc','f1','sens','spec']}
    for _ in range(n_boot):
        idx = rng.randint(0,n,size=n); yt,yp = true[idx],probs[idx]
        if len(np.unique(yt))<2: continue
        m = compute_metrics(yt,yp,nc)
        for k in res:
            if not np.isnan(m[k]): res[k].append(m[k])
    def ci(a): a=[x for x in a if not np.isnan(x)]; return (float(np.percentile(a,2.5)),float(np.percentile(a,97.5))) if a else (0,0)
    out = {}
    for k in res: lo,hi=ci(res[k]); out[f'{k}_lo']=lo; out[f'{k}_hi']=hi
    return out

def boot_roc_band(y_true, y_score, n_boot=500):
    rng = np.random.RandomState(SEED); n=len(y_true)
    mean_fpr = np.linspace(0,1,100); tprs=[]
    for _ in range(n_boot):
        idx=rng.randint(0,n,size=n); yt,ys=y_true[idx],y_score[idx]
        if len(np.unique(yt))<2: continue
        fpr,tpr,_=roc_curve(yt,ys); tprs.append(np.interp(mean_fpr,fpr,tpr))
    if not tprs: return None,None,None
    tprs=np.array(tprs)
    return mean_fpr, np.percentile(tprs,2.5,axis=0), np.percentile(tprs,97.5,axis=0)

def fmt(v,lo,hi): return f'{v:.3f} [{lo:.3f}-{hi:.3f}]'

pats = collect()
doc_pts = json.load(open(os.path.join(RES, 'doctor_roc_points.json')))
bs_df = pd.read_csv(os.path.join(TBL, 'bootstrap_full_metrics.csv'))
doc_df = pd.read_csv(os.path.join(TBL, 'doctor_individual_metrics.csv'))

# ═════════════════════════════════════════════════════════════
# STEP 1: Bootstrap CI for new face binary models
# ═════════════════════════════════════════════════════════════
print('[1] Bootstrap CI for new face binary models...')
bin_grades = {pid:(0 if p['cat']=='normal' else 1) for pid,p in pats.items()}
_,va_bin = split(bin_grades)
va_bin_l = [{'faces':pats[pid]['faces'],'label':bin_grades[pid],'pid':pid} for pid in va_bin if pid in pats]
y_bin = np.array([p['label'] for p in va_bin_l])

NEW_BIN = [
    ('Final-ConvNeXt','convnext_tiny','final_binary_face_convnext.pt'),
    ('Final-ViT','vit_tiny_patch16_224','final_binary_face_vit.pt'),
    ('Final-EffNet','efficientnet_b0','final_binary_face_efficientnet.pt'),
    ('Final-Swin','swin_tiny_patch4_window7_224','final_binary_face_swin.pt'),
]
new_bin_preds = {}; new_bin_results = []
for name,bb,ckpt in NEW_BIN:
    m = load_m(bb,ckpt,2)
    if m is None: print(f'  SKIP {name}'); continue
    probs = []
    for p in tqdm(va_bin_l, desc=name, leave=False):
        pr = predict(m, p['faces'])
        probs.append(pr if pr is not None else np.array([.5,.5]))
    probs = np.array(probs); new_bin_preds[name] = probs
    del m; torch.cuda.empty_cache()
    mt = compute_metrics(y_bin, probs); ci = boot_ci(y_bin, probs)
    r = {'name':name,'task':'Binary-Screening','nc':2,'n_val':len(y_bin),**mt,**ci}
    new_bin_results.append(r)
    print(f'  {name}: AUC={fmt(mt["auc"],ci["auc_lo"],ci["auc_hi"])} Sens={mt["sens"]:.3f} Spec={mt["spec"]:.3f}')

# Ensemble
if len(new_bin_preds)>=2:
    ens = np.mean(list(new_bin_preds.values()),axis=0)
    mt=compute_metrics(y_bin,ens); ci=boot_ci(y_bin,ens)
    new_bin_results.append({'name':'Final-Ensemble','task':'Binary-Screening','nc':2,'n_val':len(y_bin),**mt,**ci})
    new_bin_preds['Ensemble'] = ens
    print(f'  Ensemble: AUC={fmt(mt["auc"],ci["auc_lo"],ci["auc_hi"])}')

# Merge with existing bootstrap
all_bs = pd.concat([bs_df, pd.DataFrame(new_bin_results)], ignore_index=True)
all_bs.to_csv(os.path.join(TBL, 'bootstrap_full_metrics_v2.csv'), index=False, encoding='utf-8-sig')

# Save new binary results JSON
json.dump({r['name']: r['auc'] for r in new_bin_results},
          open(os.path.join(RES,'final_binary_results.json'),'w'), indent=2)

# ── Ternary predictions ──────────────────────────────────────
print('\n[2] Loading ternary predictions...')
tern_grades = {pid:{'mild':0,'moderate':1,'severe':2}[p['cat']] for pid,p in pats.items() if p['cat']!='normal'}
_,va_tern = split(tern_grades)
va_tern_l = [{'faces':pats[pid]['faces'],'label':tern_grades[pid],'pid':pid} for pid in va_tern if pid in pats]
va_eye_l = [{'eyelids':pats[pid].get('eyelids',[]),'label':tern_grades[pid],'pid':pid}
            for pid in va_tern if pid in pats and pats[pid].get('eyelids')]

TERN_EYE = [('Eyelid-ConvNeXt','convnext_tiny','eyelid_opt_convnext_tiny.pt'),
            ('Eyelid-Swin','swin_tiny_patch4_window7_224','eyelid_opt_swin_tiny_patch4_window7_224.pt')]
TERN_FACE = [('Face-ConvNeXt','convnext_tiny','v3_ternary_convnext_tiny.pt'),
             ('Face-ViT','vit_tiny_patch16_224','v3_ternary_vit_tiny_patch16_224.pt')]
tern_eye = {}; tern_face = {}
for name,bb,ckpt in TERN_EYE:
    m=load_m(bb,ckpt,3)
    if m is None: continue
    probs=[]
    for p in tqdm(va_eye_l,desc=name,leave=False):
        pr=predict(m,p['eyelids']); probs.append(pr if pr is not None else np.array([1/3]*3))
    tern_eye[name]=(np.array(probs),np.array([p['label'] for p in va_eye_l]))
    del m; torch.cuda.empty_cache()
for name,bb,ckpt in TERN_FACE:
    m=load_m(bb,ckpt,3)
    if m is None: continue
    probs=[]
    for p in tqdm(va_tern_l,desc=name,leave=False):
        pr=predict(m,p['faces']); probs.append(pr if pr is not None else np.array([1/3]*3))
    tern_face[name]=(np.array(probs),np.array([p['label'] for p in va_tern_l]))
    del m; torch.cuda.empty_cache()

# YOLO ternary
from ultralytics import YOLO
if os.path.exists(os.path.join(MODEL,'v3_yolo_ternary.pt')):
    ym=YOLO(os.path.join(MODEL,'v3_yolo_ternary.pt'))
    probs=[]
    for p in va_tern_l:
        fs=[f for f in p['faces'][:12] if os.path.exists(f)]; ps=[]
        for fp in fs:
            r=ym.predict(fp,verbose=False)
            if r and len(r)>0: ps.append(r[0].probs.data.cpu().numpy())
        probs.append(np.mean(ps,axis=0) if ps else [1/3]*3)
    tern_face['YOLO']=(np.array(probs),np.array([p['label'] for p in va_tern_l]))

all_tern = {}
for n,(p,t) in tern_eye.items(): all_tern[n]=(p,t)
for n,(p,t) in tern_face.items(): all_tern[n]=(p,t)
y_tern = np.array([p['label'] for p in va_tern_l])

# Best ternary
best_tn = max(all_tern, key=lambda n: roc_auc_score(all_tern[n][1],all_tern[n][0],multi_class='ovr',labels=[0,1,2])
              if len(np.unique(all_tern[n][1]))>1 else 0)
bp,bt = all_tern[best_tn]

# ═════════════════════════════════════════════════════════════
# STEP 3: GENERATE ALL FIGURES (English-only)
# ═════════════════════════════════════════════════════════════
print('\n[3] Generating figures (English-only, no CJK garbled text)...')

# ── FIGURE 2: Binary Screening (9 panels) ────────────────────
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.35, left=0.05, right=0.97, top=0.96, bottom=0.04)

# a: ROC with CI band + doctor points
ax = fig.add_subplot(gs[0, :2])
ax.plot([0,1],[0,1],'k--',lw=0.3,alpha=0.2)
best_bn = max(new_bin_preds, key=lambda n: roc_auc_score(y_bin, new_bin_preds[n][:,1]))
for name, probs in sorted(new_bin_preds.items(), key=lambda x: -roc_auc_score(y_bin, x[1][:,1])):
    fpr,tpr,_ = roc_curve(y_bin, probs[:,1]); auc_v = roc_auc_score(y_bin, probs[:,1])
    lw = 2.0 if name == best_bn else 1.0; alpha = 1.0 if name==best_bn else 0.6
    color = C.get('vit','#2A9D8F') if 'ViT' in name else C.get('convnext','#264653') if 'Conv' in name else C.get('effnet','#E9C46A') if 'Eff' in name else C.get('swin','#E76F51')
    ax.plot(fpr, tpr, lw=lw, alpha=alpha, color=color, label=f'{name} ({auc_v:.3f})')
fpr_b,lo,hi = boot_roc_band(y_bin, new_bin_preds[best_bn][:,1])
if fpr_b is not None: ax.fill_between(fpr_b, lo, hi, alpha=0.12, color=C['model'])
for fpr_d,tpr_d,dn,_ in doc_pts['binary']:
    ax.scatter(fpr_d, tpr_d, marker='^', s=35, c=C['doctor'], edgecolors='white', lw=0.3, zorder=5, alpha=0.6)
ax.set_xlabel('False Positive Rate', fontsize=8); ax.set_ylabel('True Positive Rate', fontsize=8)
ax.set_title('a | ROC Curves - Binary Screening', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=5, loc='lower right'); ax.set_xlim([-0.02,0.6]); ax.set_ylim([0.4,1.02])

# b: PR curves
ax = fig.add_subplot(gs[0, 2])
for name,probs in new_bin_preds.items():
    prec,rec,_ = precision_recall_curve(y_bin, probs[:,1]); ap = average_precision_score(y_bin, probs[:,1])
    color = C['vit'] if 'ViT' in name else C['convnext'] if 'Conv' in name else C['effnet'] if 'Eff' in name else C['swin']
    lw = 1.5 if name==best_bn else 1.0
    ax.plot(rec, prec, lw=lw, color=color, label=f'{name} ({ap:.3f})')
ax.set_xlabel('Recall (Sensitivity)', fontsize=8); ax.set_ylabel('Precision (PPV)', fontsize=8)
ax.set_title('b | Precision-Recall Curves', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=5, loc='lower left')

# c: Confusion matrix
ax = fig.add_subplot(gs[0, 3])
bp_bin = new_bin_preds[best_bn]; cm = confusion_matrix(y_bin, bp_bin.argmax(1))
cm_n = cm.astype(float)/cm.sum(1,keepdims=True)
im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        clr = 'white' if cm_n[i,j]>0.5 else 'black'
        ax.text(j, i, f'{cm[i,j]}\n({cm_n[i,j]:.1%})', ha='center', va='center', fontsize=8, color=clr)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Normal','Jaundiced']); ax.set_yticklabels(['Normal','Jaundiced'])
ax.set_xlabel('Predicted', fontsize=8); ax.set_ylabel('Actual', fontsize=8)
ax.set_title(f'c | Confusion Matrix ({best_bn})', fontsize=9, fontweight='bold', loc='left')

# d: Metrics bar chart
ax = fig.add_subplot(gs[1, :2])
metrics_map = {'AUC':'auc','F1':'f1','Sens':'sens','Spec':'spec','Acc':'acc','AP':'ap'}
x = np.arange(len(metrics_map)); width = 0.13
for i, name in enumerate(sorted(new_bin_preds.keys())):
    r = [rr for rr in new_bin_results if rr['name']==name]
    if not r: continue
    r = r[0]
    vals = [r[m] for m in metrics_map.values()]
    ax.bar(x + i*width - width*2, vals, width, label=name, edgecolor='white', lw=0.2)
doc_bin = doc_df[doc_df['task']=='Binary']
doc_vals = [doc_bin[m].mean() for m in metrics_map.values()]
ax.bar(x + len(new_bin_preds)*width - width*2, doc_vals, width, label='Clinicians', color=C['doctor'], hatch='//', edgecolor='white', lw=0.2)
ax.set_xticks(x); ax.set_xticklabels(list(metrics_map.keys()), fontsize=7)
ax.set_ylabel('Score', fontsize=8); ax.set_ylim([0,1.15])
ax.set_title('d | Metrics Comparison (Models + Clinicians)', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=5, ncol=3)

# e: Calibration
ax = fig.add_subplot(gs[1, 2])
frac_pos, mean_pred = calibration_curve(y_bin, bp_bin[:,1], n_bins=8, strategy='quantile')
ax.plot([0,1],[0,1],'k--',lw=0.3,alpha=0.3,label='Perfect')
ax.plot(mean_pred, frac_pos, 's-', lw=1.2, ms=4, color=C['model'], label=f'{best_bn}')
brier = brier_score_loss(y_bin, bp_bin[:,1])
ax.text(0.05, 0.85, f'Brier score: {brier:.4f}', transform=ax.transAxes, fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('Mean Predicted Probability', fontsize=8); ax.set_ylabel('Fraction of Positives', fontsize=8)
ax.set_title('e | Calibration Curve', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=6)

# f: Radar chart
ax = fig.add_subplot(gs[1, 3], polar=True)
m_labels = ['AUC','Acc','F1','Sens','Spec','AP']
best_r = [r for r in new_bin_results if r['name']==best_bn or (best_bn=='Ensemble' and r['name']=='Final-Ensemble')]
best_r = best_r[0] if best_r else new_bin_results[0]
model_v = [best_r[m] for m in ['auc','acc','f1','sens','spec','ap']]
doc_v = [doc_bin[k].mean() for k in ['auc','acc','f1','sens','spec','ap']]
ang = np.linspace(0,2*np.pi,len(m_labels),endpoint=False).tolist(); ang+=ang[:1]
model_v+=model_v[:1]; doc_v+=doc_v[:1]
ax.plot(ang, model_v, 'o-', lw=1.2, ms=3, color=C['model'], label='BilinGuard')
ax.fill(ang, model_v, alpha=0.1, color=C['model'])
ax.plot(ang, doc_v, 's-', lw=1.2, ms=3, color=C['doctor'], label='Clinicians')
ax.fill(ang, doc_v, alpha=0.1, color=C['doctor'])
ax.set_xticks(ang[:-1]); ax.set_xticklabels(m_labels, fontsize=6)
ax.set_ylim([0,1.05]); ax.set_title('f | Radar Comparison', fontsize=9, fontweight='bold', loc='left', pad=12)
ax.legend(fontsize=5, loc='upper right', bbox_to_anchor=(1.35,1.1))

# g: Score distribution
ax = fig.add_subplot(gs[2, :2])
ax.hist(bp_bin[y_bin==0,1], bins=20, alpha=0.5, color=C['vit'], label=f'Normal (n={sum(y_bin==0)})', density=True)
ax.hist(bp_bin[y_bin==1,1], bins=20, alpha=0.5, color=C['swin'], label=f'Jaundiced (n={sum(y_bin==1)})', density=True)
ax.axvline(x=0.5, color='red', ls='--', lw=0.8, label='Threshold=0.5')
ax.set_xlabel('Predicted Probability of Jaundice', fontsize=8); ax.set_ylabel('Density', fontsize=8)
ax.set_title(f'g | Score Distribution ({best_bn})', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=6)

# h: AUC ranking with CI
ax = fig.add_subplot(gs[2, 2])
for i,(_,r) in enumerate(pd.DataFrame(new_bin_results).sort_values('auc').iterrows()):
    ax.errorbar(r['auc'], i, xerr=[[r['auc']-r['auc_lo']],[r['auc_hi']-r['auc']]], fmt='o', ms=4, capsize=2, color=C['model'])
ax.set_yticks(range(len(new_bin_results)))
ax.set_yticklabels(pd.DataFrame(new_bin_results).sort_values('auc')['name'].tolist(), fontsize=6)
ax.set_xlabel('AUC-ROC [95% CI]', fontsize=8)
ax.set_title('h | AUC Ranking with CI', fontsize=9, fontweight='bold', loc='left')
ax.axvline(0.5, color='gray', ls='--', lw=0.3)

# i: Operating point analysis
ax = fig.add_subplot(gs[2, 3])
thresholds = np.linspace(0.01,0.99,50); sens_a,spec_a=[],[]
for t in thresholds:
    pred_t = (bp_bin[:,1]>=t).astype(int); cm_t = confusion_matrix(y_bin, pred_t, labels=[0,1])
    sens_a.append(cm_t[1,1]/max(cm_t[1,:].sum(),1)); spec_a.append(cm_t[0,0]/max(cm_t[0,:].sum(),1))
ax.plot(1-np.array(spec_a), sens_a, lw=1.2, color=C['model'])
idx05 = np.argmin(np.abs(thresholds-0.5))
ax.scatter(1-spec_a[idx05], sens_a[idx05], s=50, c='red', zorder=5, label='Threshold=0.5')
ax.set_xlabel('1 - Specificity (FPR)', fontsize=8); ax.set_ylabel('Sensitivity (TPR)', fontsize=8)
ax.set_title('i | Operating Point Analysis', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=6)

fig.savefig(os.path.join(FIG, 'Figure2_binary_screening.png'))
fig.savefig(os.path.join(FIG, 'Figure2_binary_screening.svg'))
plt.close(fig)
print('  Figure 2 done')

# ── FIGURE 3: Ternary (10 panels) ────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.35, left=0.05, right=0.97, top=0.96, bottom=0.04)

# a: Micro-avg ROC
ax = fig.add_subplot(gs[0, :2])
ax.plot([0,1],[0,1],'k--',lw=0.3,alpha=0.2)
for name,(probs,true) in sorted(all_tern.items(), key=lambda x: -roc_auc_score(x[1][1],x[1][0],multi_class='ovr',labels=[0,1,2])):
    yb = label_binarize(true, classes=[0,1,2]); fpr,tpr,_ = roc_curve(yb.ravel(), probs.ravel())
    from sklearn.metrics import auc as sk_auc; a = sk_auc(fpr,tpr)
    color = C['convnext'] if 'Conv' in name else C['swin'] if 'Swin' in name else C['vit'] if 'ViT' in name else C['yolo']
    lw = 2.0 if name==best_tn else 1.0
    ax.plot(fpr, tpr, lw=lw, color=color, label=f'{name} ({a:.3f})')
for fpr_d,tpr_d,dn,_ in doc_pts.get('ternary',[]):
    ax.scatter(fpr_d, tpr_d, marker='^', s=35, c=C['doctor'], edgecolors='white', lw=0.3, zorder=5, alpha=0.6)
ax.set_xlabel('False Positive Rate', fontsize=8); ax.set_ylabel('True Positive Rate', fontsize=8)
ax.set_title('a | Micro-averaged ROC - Ternary Grading', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=5, loc='lower right'); ax.set_xlim([-0.02,0.6]); ax.set_ylim([0.2,1.02])

# b: PR
ax = fig.add_subplot(gs[0, 2])
for name,(probs,true) in all_tern.items():
    yb = label_binarize(true, classes=[0,1,2])
    prec,rec,_ = precision_recall_curve(yb.ravel(), probs.ravel())
    ap = average_precision_score(yb, probs, average='micro')
    color = C['convnext'] if 'Conv' in name else C['swin'] if 'Swin' in name else C['vit'] if 'ViT' in name else C['yolo']
    lw = 1.5 if name==best_tn else 1.0
    ax.plot(rec, prec, lw=lw, color=color, label=f'{name} ({ap:.3f})')
ax.set_xlabel('Recall', fontsize=8); ax.set_ylabel('Precision', fontsize=8)
ax.set_title('b | Precision-Recall Curves', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=5, loc='lower left')

# c: Confusion matrix
ax = fig.add_subplot(gs[0, 3])
cm = confusion_matrix(bt, bp.argmax(1), labels=[0,1,2]); cm_n = cm.astype(float)/cm.sum(1,keepdims=True)
im = ax.imshow(cm_n, cmap='Greens', vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        clr = 'white' if cm_n[i,j]>0.5 else 'black'
        ax.text(j,i,f'{cm[i,j]}\n({cm_n[i,j]:.1%})', ha='center', va='center', fontsize=7, color=clr)
ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
ax.set_xticklabels(['Mild','Moderate','Severe']); ax.set_yticklabels(['Mild','Moderate','Severe'])
ax.set_xlabel('Predicted', fontsize=8); ax.set_ylabel('Actual', fontsize=8)
ax.set_title(f'c | Confusion Matrix ({best_tn})', fontsize=9, fontweight='bold', loc='left')

# d-f: Per-class ROC
for idx,(cls,cname,ccolor) in enumerate([(0,'Mild',C['mild']),(1,'Moderate',C['moderate']),(2,'Severe',C['severe'])]):
    ax = fig.add_subplot(gs[1, idx])
    yb_c = (bt==cls).astype(int); ax.plot([0,1],[0,1],'k--',lw=0.3,alpha=0.2)
    fpr,tpr,_ = roc_curve(yb_c, bp[:,cls]); auc_c = roc_auc_score(yb_c, bp[:,cls])
    ax.plot(fpr, tpr, lw=1.5, color=ccolor, label=f'AUC={auc_c:.3f}')
    fpr_b,lo,hi = boot_roc_band(yb_c, bp[:,cls])
    if fpr_b is not None: ax.fill_between(fpr_b, lo, hi, alpha=0.12, color=ccolor)
    ax.set_xlabel('FPR', fontsize=7); ax.set_ylabel('TPR', fontsize=7)
    ax.set_title(f'{chr(100+idx)} | {cname} vs Rest (OvR)', fontsize=8, fontweight='bold', loc='left')
    ax.legend(fontsize=6)

# g: Calibration per class
ax = fig.add_subplot(gs[1, 3])
for cls,cname,ccolor in [(0,'Mild',C['mild']),(1,'Moderate',C['moderate']),(2,'Severe',C['severe'])]:
    yb_c = (bt==cls).astype(int)
    if sum(yb_c)>3:
        frac,mean_p = calibration_curve(yb_c, bp[:,cls], n_bins=5, strategy='quantile')
        ax.plot(mean_p, frac, 's-', lw=1, ms=3, color=ccolor, label=cname)
ax.plot([0,1],[0,1],'k--',lw=0.3,alpha=0.3)
ax.set_xlabel('Mean Predicted Probability', fontsize=8); ax.set_ylabel('Fraction of Positives', fontsize=8)
ax.set_title('g | Calibration (Per-Class)', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=6)

# h: AUC ranking
ax = fig.add_subplot(gs[2, :2])
tern_bs = pd.concat([bs_df[bs_df['task']=='TBIL-Eyelid'], bs_df[bs_df['task']=='Face-Ternary']]).sort_values('auc')
for i,(_,r) in enumerate(tern_bs.iterrows()):
    color = C['model'] if 'Eyelid' in r['name'] else C['vit']
    ax.errorbar(r['auc'], i, xerr=[[r['auc']-r['auc_lo']],[r['auc_hi']-r['auc']]], fmt='o', ms=4, capsize=2, color=color)
doc_tern = doc_df[doc_df['task']=='Ternary']
ax.axvline(doc_tern['auc'].mean(), color=C['doctor'], ls='--', lw=1, label=f'Clinicians mean ({doc_tern["auc"].mean():.3f})')
ax.set_yticks(range(len(tern_bs))); ax.set_yticklabels(tern_bs['name'].tolist(), fontsize=5)
ax.set_xlabel('AUC-ROC [95% CI]', fontsize=8)
ax.set_title('h | Ternary AUC Ranking with CI', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=6); ax.axvline(0.5, color='gray', ls='--', lw=0.3)

# i: Per-class sens/spec
ax = fig.add_subplot(gs[2, 2])
cm = confusion_matrix(bt, bp.argmax(1), labels=[0,1,2]); sens_c=[]; spec_c=[]
for c in range(3):
    tp=cm[c,c];fn=cm[c,:].sum()-tp;fp=cm[:,c].sum()-tp;tn=cm.sum()-tp-fn-fp
    sens_c.append(tp/max(tp+fn,1)); spec_c.append(tn/max(tn+fp,1))
x=np.arange(3); w=0.3
ax.bar(x-w/2, sens_c, w, color=[C['mild'],C['moderate'],C['severe']], alpha=0.7, label='Sensitivity')
ax.bar(x+w/2, spec_c, w, color=[C['mild'],C['moderate'],C['severe']], alpha=0.3, label='Specificity', hatch='//')
ax.set_xticks(x); ax.set_xticklabels(['Mild','Moderate','Severe'], fontsize=7)
ax.set_ylabel('Score', fontsize=8); ax.set_title('i | Per-Class Sens/Spec', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=6)

# j: Model vs Doctor
ax = fig.add_subplot(gs[2, 3])
metrics_t = ['auc','f1','sens','spec']; m_labels_t = ['AUC','F1','Sens','Spec']
best_tbs = tern_bs.nlargest(1,'auc').iloc[0]
model_v = [best_tbs[m] for m in metrics_t]
doc_v = [doc_tern[m].mean() for m in metrics_t]; doc_s = [doc_tern[m].std() for m in metrics_t]
x=np.arange(4); w=0.3
ax.bar(x-w/2, model_v, w, color=C['model'], label='BilinGuard')
ax.bar(x+w/2, doc_v, w, yerr=doc_s, color=C['doctor'], capsize=2, label='Clinicians')
ax.set_xticks(x); ax.set_xticklabels(m_labels_t, fontsize=7)
ax.set_ylabel('Score', fontsize=8); ax.set_title('j | Model vs Clinician', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=6)

fig.savefig(os.path.join(FIG, 'Figure3_ternary_grading.png'))
fig.savefig(os.path.join(FIG, 'Figure3_ternary_grading.svg'))
plt.close(fig)
print('  Figure 3 done')

# ── FIGURE 4: Model vs Clinician (8 panels) ──────────────────
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35, left=0.05, right=0.96, top=0.96, bottom=0.04)

# a: Binary ROC with CI band + doctor points
ax = fig.add_subplot(gs[0, :2])
ym = new_bin_preds[best_bn][:,1]
fpr_m,tpr_m,_ = roc_curve(y_bin, ym); auc_m = roc_auc_score(y_bin, ym)
ax.plot(fpr_m, tpr_m, lw=2, color=C['model'], label=f'BilinGuard ({auc_m:.3f})')
fpr_b,lo,hi = boot_roc_band(y_bin, ym)
if fpr_b is not None: ax.fill_between(fpr_b, lo, hi, alpha=0.15, color=C['model'])
ax.plot([0,1],[0,1],'k--',lw=0.3,alpha=0.2)
for fpr_d,tpr_d,dn,auc_d in doc_pts['binary']:
    short = dn.split('-')[-1][:6]
    ax.scatter(fpr_d, tpr_d, marker='^', s=50, c=C['doctor'], edgecolors=C['model'], lw=0.3, zorder=5)
    ax.annotate(short, (fpr_d, tpr_d), fontsize=4.5, textcoords='offset points', xytext=(4,4))
doc_fpr = 1-doc_bin['spec'].mean(); doc_tpr = doc_bin['sens'].mean()
ax.scatter(doc_fpr, doc_tpr, marker='*', s=150, c=C['doctor'], edgecolors='black', lw=0.5, zorder=6, label=f'Clinician mean ({doc_bin["auc"].mean():.3f})')
ax.set_xlabel('False Positive Rate', fontsize=8); ax.set_ylabel('True Positive Rate', fontsize=8)
ax.set_title('a | Binary: BilinGuard vs Clinicians', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=6); ax.set_xlim([-0.02,0.55]); ax.set_ylim([0.4,1.02])

# b: Ternary ROC + doctor points
ax = fig.add_subplot(gs[0, 2])
yb = label_binarize(bt, classes=[0,1,2])
fpr_m,tpr_m,_ = roc_curve(yb.ravel(), bp.ravel())
from sklearn.metrics import auc as sk_auc; auc_m = sk_auc(fpr_m, tpr_m)
ax.plot(fpr_m, tpr_m, lw=2, color=C['model'], label=f'BilinGuard ({auc_m:.3f})')
ax.plot([0,1],[0,1],'k--',lw=0.3,alpha=0.2)
for fpr_d,tpr_d,dn,_ in doc_pts.get('ternary',[]):
    ax.scatter(fpr_d, tpr_d, marker='^', s=50, c=C['doctor'], edgecolors=C['model'], lw=0.3, zorder=5)
ax.set_xlabel('FPR', fontsize=8); ax.set_ylabel('TPR', fontsize=8)
ax.set_title('b | Ternary: BilinGuard vs Clinicians', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=6); ax.set_xlim([-0.02,0.55]); ax.set_ylim([0.1,1.02])

# c: Binary F1 ranking
ax = fig.add_subplot(gs[1, 0])
best_r = [r for r in new_bin_results if r['name']==best_bn or (best_bn=='Ensemble' and r['name']=='Final-Ensemble')]
best_r = best_r[0] if best_r else new_bin_results[0]
all_f1 = doc_bin.sort_values('f1')['f1'].tolist() + [best_r['f1']]
all_names = doc_bin.sort_values('f1')['doctor'].tolist() + ['BilinGuard']
colors_b = [C['doctor']]*len(doc_bin) + [C['model']]
ax.barh(range(len(all_f1)), all_f1, color=colors_b, edgecolor='white', lw=0.2)
ax.set_yticks(range(len(all_f1))); ax.set_yticklabels([n.split('-')[-1][:8] for n in all_names], fontsize=5)
ax.set_xlabel('F1 Score', fontsize=8); ax.set_title('c | Binary F1 Ranking', fontsize=9, fontweight='bold', loc='left')

# d: Ternary F1 ranking
ax = fig.add_subplot(gs[1, 1])
all_f1_t = doc_tern.sort_values('f1')['f1'].tolist() + [best_tbs['f1']]
all_names_t = doc_tern.sort_values('f1')['doctor'].tolist() + ['BilinGuard']
colors_t = [C['doctor']]*len(doc_tern) + [C['model']]
ax.barh(range(len(all_f1_t)), all_f1_t, color=colors_t, edgecolor='white', lw=0.2)
ax.set_yticks(range(len(all_f1_t))); ax.set_yticklabels([n.split('-')[-1][:8] for n in all_names_t], fontsize=5)
ax.set_xlabel('F1 Score', fontsize=8); ax.set_title('d | Ternary F1 Ranking', fontsize=9, fontweight='bold', loc='left')

# e: Response time
ax = fig.add_subplot(gs[1, 2])
doc_dir = os.path.join(BASE, 'doctor'); doc_times = {}
for fname in os.listdir(doc_dir):
    if not fname.endswith('.xlsx') or any(x in fname for x in ['sum','agg','perf']): continue
    df = pd.read_excel(os.path.join(doc_dir, fname))
    if df.shape[1]>1:
        times = pd.to_numeric(df.iloc[:,1], errors='coerce').dropna()
        if len(times)>50: doc_times[fname.replace('.xlsx','').replace('-','')[-6:]] = times.values
bp_res = ax.boxplot(list(doc_times.values()), tick_labels=list(doc_times.keys()), patch_artist=True, widths=0.5)
for patch in bp_res['boxes']: patch.set_facecolor(C['doctor']); patch.set_alpha(0.5)
ax.axhline(y=0.5, color=C['model'], ls='--', lw=1.5, label='BilinGuard (~0.5s)')
ax.set_ylabel('Response Time (seconds)', fontsize=8)
ax.set_title('e | Time Efficiency', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=6); plt.setp(ax.get_xticklabels(), rotation=30, fontsize=5, ha='right')

# f: Wilcoxon P-values
ax = fig.add_subplot(gs[2, 0])
reader = json.load(open(os.path.join(RES, 'reader_study_results.json')))
if 'wilcoxon' in reader:
    pvals = [reader['wilcoxon'][k]['p_value'] for k in sorted(reader['wilcoxon'].keys())]
    labels = [k.replace('_Attending','').replace('_Resident','').replace('_Nurse','').replace('_PubHealth','') for k in sorted(reader['wilcoxon'].keys())]
    log_p = [-np.log10(max(p,1e-10)) for p in pvals]
    colors_p = ['#E74C3C' if p<0.001 else '#E67E22' if p<0.01 else '#F4D03F' if p<0.05 else '#2A9D8F' for p in pvals]
    ax.barh(range(len(log_p)), log_p, color=colors_p, edgecolor='white', lw=0.2)
    ax.set_yticks(range(len(log_p))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel('-log10(P value)', fontsize=8)
    ax.axvline(-np.log10(0.05), color='gray', ls='--', lw=0.5, label='P=0.05')
    ax.axvline(-np.log10(0.001), color='red', ls='--', lw=0.5, label='P=0.001')
    ax.set_title('f | Wilcoxon P-values', fontsize=9, fontweight='bold', loc='left')
    ax.legend(fontsize=5)

# g: Binary all metrics
ax = fig.add_subplot(gs[2, 1])
metrics_c = ['auc','f1','sens','spec','ap']; m_lab = ['AUC','F1','Sens','Spec','AP']
model_v = [best_r[m] for m in metrics_c]; doc_v = [doc_bin[m].mean() for m in metrics_c]
doc_e = [doc_bin[m].std() for m in metrics_c]; x=np.arange(5); w=0.3
ax.bar(x-w/2, model_v, w, color=C['model'], label='BilinGuard')
ax.bar(x+w/2, doc_v, w, yerr=doc_e, color=C['doctor'], capsize=2, label='Clinicians')
ax.set_xticks(x); ax.set_xticklabels(m_lab, fontsize=7); ax.set_ylim([0,1.15])
ax.set_title('g | Binary: All Metrics', fontsize=9, fontweight='bold', loc='left'); ax.legend(fontsize=6)

# h: Ternary all metrics
ax = fig.add_subplot(gs[2, 2])
model_v = [best_tbs[m] for m in metrics_c]; doc_v = [doc_tern[m].mean() for m in metrics_c]
doc_e = [doc_tern[m].std() for m in metrics_c]
ax.bar(x-w/2, model_v, w, color=C['model'], label='BilinGuard')
ax.bar(x+w/2, doc_v, w, yerr=doc_e, color=C['doctor'], capsize=2, label='Clinicians')
ax.set_xticks(x); ax.set_xticklabels(m_lab, fontsize=7); ax.set_ylim([0,1.15])
ax.set_title('h | Ternary: All Metrics', fontsize=9, fontweight='bold', loc='left'); ax.legend(fontsize=6)

fig.savefig(os.path.join(FIG, 'Figure4_model_vs_clinician.png'))
fig.savefig(os.path.join(FIG, 'Figure4_model_vs_clinician.svg'))
plt.close(fig)
print('  Figure 4 done')

# ── SUPPLEMENTARY FIGURES ────────────────────────────────────
print('  Supplementary figures...')

# S1: External validation
ext_df = pd.read_csv(os.path.join(TBL, 'external_validation_results.csv'))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]; es = ext_df.sort_values('AUC')
ax.barh(range(len(es)), es['AUC'], color=[C['model'] if 'YOLO' in n else C['vit'] for n in es['Algorithm']])
ax.set_yticks(range(len(es))); ax.set_yticklabels(es['Algorithm'], fontsize=6)
ax.set_xlabel('AUC-ROC', fontsize=9); ax.set_title('External Validation AUC', fontsize=10, fontweight='bold')
for i,v in enumerate(es['AUC']): ax.text(v+0.01, i, f'{v:.3f}', va='center', fontsize=6)
ax = axes[1]
for _,r in es.iterrows():
    if r['AUC']>0.5:
        ax.bar(r['Algorithm'][:10], r['F1'], color=C['vit'], alpha=0.6)
        ax.bar(r['Algorithm'][:10], r['Accuracy'], color=C['model'], alpha=0.3)
ax.set_ylabel('Score'); ax.set_title('External: F1 (solid) vs Accuracy (hatch)', fontsize=10, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=30, fontsize=6, ha='right')
plt.tight_layout()
fig.savefig(os.path.join(SUPP, 'SuppFig1_external_validation.png')); fig.savefig(os.path.join(SUPP, 'SuppFig1_external_validation.svg'))
plt.close(fig)

# S2: Forest plots (all tasks)
fig, axes = plt.subplots(2, 3, figsize=(18, 14))
for ax,task in zip(axes.ravel(), sorted(all_bs['task'].unique())):
    sub = all_bs[all_bs['task']==task].sort_values('auc')
    for i,(_,r) in enumerate(sub.iterrows()):
        ax.errorbar(r['auc'], i, xerr=[[r['auc']-r['auc_lo']],[r['auc_hi']-r['auc']]], fmt='o', ms=4, capsize=2, color=C['model'])
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub['name'].tolist(), fontsize=5)
    ax.set_xlabel('AUC-ROC [95% CI]', fontsize=7); ax.set_title(task, fontsize=8, fontweight='bold')
    ax.axvline(0.5, color='gray', ls='--', lw=0.3)
plt.suptitle('Bootstrap 95% CI Forest Plots (1000 iterations)', fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.97])
fig.savefig(os.path.join(SUPP, 'SuppFig2_forest_plots.png')); fig.savefig(os.path.join(SUPP, 'SuppFig2_forest_plots.svg'))
plt.close(fig)

# S3: DBIL/IBIL
dbil = json.load(open(os.path.join(RES,'dbil_classification_results.json')))
ibil = json.load(open(os.path.join(RES,'ibil_classification_results.json')))
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax,data,title in [(axes[0,0],dbil,'DBIL'),(axes[0,1],ibil,'IBIL')]:
    items = sorted(data.items(), key=lambda x:-x[1])
    names = [k.split('_')[-1].title() if 'ensemble' not in k else 'Ensemble' for k,_ in items]
    vals = [v for _,v in items]
    colors_b = [C['convnext'] if 'convnext' in k else C['vit'] if 'vit' in k else C['effnet'] if 'efficientnet' in k else C['swin'] if 'swin' in k else C['yolo'] for k,_ in items]
    ax.barh(range(len(vals)), vals, color=colors_b, edgecolor='white', lw=0.2)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('AUC-ROC'); ax.set_title(f'{title} Ternary Classification', fontsize=9, fontweight='bold')
    for i,v in enumerate(vals): ax.text(v+0.01, i, f'{v:.3f}', va='center', fontsize=6)
    ax.set_xlim([0.3,1.0])
ax = axes[1,0]
for i,(data,label) in enumerate([(dbil,'DBIL'),(ibil,'IBIL')]):
    eb = max([v for k,v in data.items() if 'eyelid' in k], default=0)
    fb = max([v for k,v in data.items() if 'face' in k], default=0)
    ax.bar([i*2,i*2+1], [eb,fb], color=[C['convnext'],C['vit']], width=0.6)
ax.set_xticks([0,1,2,3]); ax.set_xticklabels(['DBIL\nEyelid','DBIL\nFace','IBIL\nEyelid','IBIL\nFace'], fontsize=7)
ax.set_ylabel('Best AUC-ROC'); ax.set_title('DBIL vs IBIL: Best per Modality', fontsize=9, fontweight='bold')
ax = axes[1,1]; ax.text(0.5,0.5,'Grading Standards:\nDBIL: <=10 / 10-68 / >68 umol/L\nIBIL: <=20 / 20-50 / >50 umol/L', ha='center', va='center', fontsize=10, transform=ax.transAxes)
ax.set_title('Bilirubin Grading Standards', fontsize=9, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(SUPP, 'SuppFig3_bilirubin_classification.png')); fig.savefig(os.path.join(SUPP, 'SuppFig3_bilirubin_classification.svg'))
plt.close(fig)

# S4: Color distributions
cdf = pd.read_csv(os.path.join(RES, 'color_stats_diagnosis.csv'))
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax,feat in zip(axes.ravel(), ['mean_R','mean_G','mean_B','Lab_b','yellow_ratio','HSV_H']):
    for cat,color in [('normal',C['vit']),('mild',C['mild']),('moderate',C['moderate']),('severe',C['severe'])]:
        vals = cdf[cdf['category']==cat][feat].dropna()
        if len(vals)>0: ax.hist(vals, bins=20, alpha=0.4, label=cat, color=color)
    ax.set_title(feat, fontsize=8); ax.legend(fontsize=5)
plt.suptitle('Color Feature Distributions by Bilirubin Category', fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(SUPP, 'SuppFig4_color_distributions.png')); fig.savefig(os.path.join(SUPP, 'SuppFig4_color_distributions.svg'))
plt.close(fig)

# S5: Response times violin
fig, ax = plt.subplots(figsize=(10, 5))
parts = ax.violinplot(list(doc_times.values()), positions=range(len(doc_times)), showmeans=True, showmedians=True)
for pc in parts['bodies']: pc.set_facecolor(C['doctor']); pc.set_alpha(0.4)
ax.set_xticks(range(len(doc_times))); ax.set_xticklabels(list(doc_times.keys()), fontsize=6, rotation=30, ha='right')
ax.set_ylabel('Response Time (seconds)', fontsize=9)
ax.axhline(y=0.5, color=C['model'], ls='--', lw=1.5, label='BilinGuard (~0.5s)')
ax.set_title('Clinician Response Time Distribution', fontsize=10, fontweight='bold')
ax.legend(fontsize=8); plt.tight_layout()
fig.savefig(os.path.join(SUPP, 'SuppFig5_response_times.png')); fig.savefig(os.path.join(SUPP, 'SuppFig5_response_times.svg'))
plt.close(fig)

# S6: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
cm = confusion_matrix(y_bin, new_bin_preds[best_bn].argmax(1))
ax = axes[0]; cm_n = cm.astype(float)/cm.sum(1,keepdims=True)
im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j,i,f'{cm[i,j]}\n({cm_n[i,j]:.1%})', ha='center', va='center', fontsize=8, color='white' if cm_n[i,j]>0.5 else 'black')
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(['Normal','Jaundiced']); ax.set_yticklabels(['Normal','Jaundiced'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(f'Binary ({best_bn})', fontsize=10, fontweight='bold')
cm = confusion_matrix(bt, bp.argmax(1), labels=[0,1,2])
ax = axes[1]; cm_n = cm.astype(float)/cm.sum(1,keepdims=True)
im = ax.imshow(cm_n, cmap='Greens', vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j,i,f'{cm[i,j]}\n({cm_n[i,j]:.1%})', ha='center', va='center', fontsize=7, color='white' if cm_n[i,j]>0.5 else 'black')
ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2]); ax.set_xticklabels(['Mild','Mod','Sev']); ax.set_yticklabels(['Mild','Mod','Sev'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(f'Ternary ({best_tn})', fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(SUPP, 'SuppFig6_confusion_matrices.png')); fig.savefig(os.path.join(SUPP, 'SuppFig6_confusion_matrices.svg'))
plt.close(fig)

print('  All supplementary figures done')

# ── TABLES ───────────────────────────────────────────────────
print('\n[4] Generating tables...')

# Manuscript table format (tab-separated, one row per model)
with open(os.path.join(TBL, 'manuscript_tables_v2.tsv'), 'w', encoding='utf-8') as f:
    for task in sorted(all_bs['task'].unique()):
        sub = all_bs[all_bs['task']==task].sort_values('auc', ascending=False)
        f.write(f'\n=== {task} ===\n')
        f.write('Model\tF1 score\tSensitivity\tSpecificity\tAccuracy\tAUC ROC\tAverage Precision\n')
        for _,r in sub.iterrows():
            f.write('\t'.join([r['name'],
                fmt(r['f1'],r['f1_lo'],r['f1_hi']),
                fmt(r['sens'],r['sens_lo'],r['sens_hi']),
                fmt(r['spec'],r['spec_lo'],r['spec_hi']),
                fmt(r['acc'],r['acc_lo'],r['acc_hi']),
                fmt(r['auc'],r['auc_lo'],r['auc_hi']),
                fmt(r['ap'],r['ap_lo'],r['ap_hi'])]) + '\n')
    # Doctor table
    f.write('\n=== Clinician Performance (Individual) ===\n')
    f.write('Task\tDoctor\tF1\tSensitivity\tSpecificity\tAccuracy\tAUC\tFPR\tTPR\n')
    for _,r in doc_df.iterrows():
        f.write(f'{r["task"]}\t{r["doctor"]}\t{r["f1"]:.4f}\t{r["sens"]:.4f}\t{r["spec"]:.4f}\t{r["acc"]:.4f}\t{r["auc"]:.4f}\t{r["fpr"]:.4f}\t{r["tpr"]:.4f}\n')

print(f'  Tables: {TBL}/manuscript_tables_v2.tsv')

# ── SUMMARY ──────────────────────────────────────────────────
print('\n=== COMPLETE ===')
print(f'\nNew binary models (strong augmentation, stable):')
for r in new_bin_results:
    print(f'  {r["name"]}: AUC={r["auc"]:.3f} [{r["auc_lo"]:.3f}-{r["auc_hi"]:.3f}]')
print(f'\nAll figures in: {FIG} and {SUPP}')
for d in [FIG, SUPP]:
    pngs = [f for f in os.listdir(d) if f.endswith('.png')]
    print(f'  {d}: {len(pngs)} PNG files')
