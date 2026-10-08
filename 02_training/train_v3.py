# -*- coding: utf-8 -*-
"""
BilinGuard v3 Training — SAM v3 data (occlusion-filtered, original pixels)
  Stage 1: Binary screening (normal vs jaundiced)
  Stage 2: Ternary grading (ViT + Swin + EfficientNet + ConvNeXt + YellowFeatures)
  Ensemble: DynamicFusion = BilinGuard
"""
import os, json, random, copy
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
DEV = torch.device('cuda')
S = 42; IS = 224; BS = 32; EP = 30; LR = 1e-4; MF = 12
random.seed(S); np.random.seed(S); torch.manual_seed(S); torch.cuda.manual_seed_all(S)
print(f'GPU: {torch.cuda.get_device_name(0)}')

class FocalLoss(nn.Module):
    def __init__(self, a=None, g=2.0, ls=0.1):
        super().__init__(); self.a=a; self.g=g; self.ls=ls
    def forward(self, lo, t):
        nc=lo.size(1); st=torch.zeros_like(lo).scatter_(1,t.unsqueeze(1),1.0)
        st=st*(1-self.ls)+self.ls/nc
        lp=F.log_softmax(lo,1); p=torch.exp(lp)
        fw=(1-p.gather(1,t.unsqueeze(1)).clamp(min=1e-8))**self.g
        l=-(st*lp).sum(1)*fw.squeeze()
        return (l*self.a[t]).mean() if self.a is not None else l.mean()

class EMA:
    def __init__(self,m,d=0.999):
        self.d=d; self.s={n:p.data.clone() for n,p in m.named_parameters() if p.requires_grad}
    def update(self,m):
        for n,p in m.named_parameters():
            if n in self.s: self.s[n]=self.d*self.s[n]+(1-self.d)*p.data
    def apply(self,m):
        b={}
        for n,p in m.named_parameters():
            if n in self.s: b[n]=p.data.clone(); p.data=self.s[n].clone()
        return b
    def restore(self,m,b):
        for n,p in m.named_parameters():
            if n in b: p.data=b[n].clone()

class YellowFeatures(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.c1=nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))
        self.c2=nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2))
        self.c3=nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(2))
        self.p=nn.AdaptiveAvgPool2d(1)
        self.f=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nc))
    def forward(self,x):
        return self.f(self.p(self.c3(self.c2(self.c1(x)))).flatten(1))

tr_tf=transforms.Compose([transforms.Resize((IS,IS)),transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.1,0.1,0.05,0.03),transforms.RandomRotation(10),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ev_tf=transforms.Compose([transforms.Resize((IS,IS)),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class DS(Dataset):
    def __init__(self,ps,key,lmap,tf,mx=MF):
        self.s=[]; self.tf=tf
        for p in ps:
            if p['cat'] not in lmap: continue
            for ip in p.get(key,[])[:mx]: self.s.append((ip,lmap[p['cat']],p['id']))
    def __len__(self): return len(self.s)
    def __getitem__(self,i):
        p,l,pi=self.s[i]
        try: im=Image.open(p).convert('RGB')
        except: im=Image.new('RGB',(IS,IS),(128,128,128))
        return self.tf(im),l,pi

def find_p(root,cats):
    ps=[]
    for c in cats:
        cd=os.path.join(root,c)
        if not os.path.exists(cd): continue
        for d in os.listdir(cd):
            p=os.path.join(cd,d)
            if not os.path.isdir(p): continue
            fs=os.listdir(p)
            faces=sorted([os.path.join(p,f) for f in fs if '_face' in f and f.endswith('.jpg')])
            scl=sorted([os.path.join(p,f) for f in fs if '_sclera' in f and f.endswith('.jpg')])
            if faces: ps.append({'id':d,'faces':faces,'scleras':scl,'cat':c})
    return ps

def pagg(ps,ls,pis):
    d={}
    for p,l,pi in zip(ps,ls,pis): d.setdefault(pi,([],l)); d[pi][0].append(p)
    pis=list(d.keys())
    return np.array([np.mean(d[p][0],0) for p in pis]),np.array([d[p][1] for p in pis])

def split(ps,fn,r=0.2):
    bl={}
    for p in ps: bl.setdefault(fn(p),[]).append(p)
    tr,va=[],[]
    for l,g in bl.items():
        random.shuffle(g); n=max(1,int(len(g)*r))
        va.extend(g[:n]); tr.extend(g[n:])
    return tr,va

def train(m,trl,vrl,nc,task,sn):
    cnt=np.bincount([l for _,l,_ in trl.dataset.s],minlength=nc)
    al=torch.FloatTensor((1./cnt)/(1./cnt).sum()).to(DEV)
    cr=FocalLoss(a=al)
    op=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4)
    sc=torch.optim.lr_scheduler.CosineAnnealingLR(op,T_max=EP)
    em=EMA(m); ba=0
    for ep in range(1,EP+1):
        m.train()
        for im,la,_ in tqdm(trl,desc=f'{task} E{ep}',leave=False):
            im,la=im.to(DEV),la.to(DEV)
            op.zero_grad(); lo=cr(m(im),la); lo.backward(); op.step(); em.update(m)
        sc.step()
        bk=em.apply(m); m.eval()
        ps,ls,pis=[],[],[]
        with torch.no_grad():
            for im,la,pi in vrl:
                o=m(im.to(DEV)); ps.extend(F.softmax(o,1).cpu().numpy()); ls.extend(la.numpy()); pis.extend(pi)
        avg,true=pagg(np.array(ps),np.array(ls),pis)
        pl=avg.argmax(1)
        try: au=roc_auc_score(true,avg[:,1]) if nc==2 else roc_auc_score(true,avg,multi_class='ovr')
        except: au=0
        ac=accuracy_score(true,pl); f1=f1_score(true,pl,average='macro')
        if ep%5==0 or ep==1: print(f'  {task} E{ep:3d}: acc={ac:.4f} f1={f1:.4f} auc={au:.4f}')
        if au>ba: ba=au; torch.save(m.state_dict(),os.path.join(MODEL,f'{sn}.pt'))
        em.restore(m,bk)
    print(f'  {task} Best: {ba:.4f}')
    del m; torch.cuda.empty_cache()
    return ba

# ── Load data ────────────────────────────────────────────────
print('\n[1] Data...')
allp=find_p(DATA,['normal','mild','moderate','severe'])
for c in ['normal','mild','moderate','severe']:
    n=sum(1 for p in allp if p['cat']==c)
    im=sum(len(p['faces']) for p in allp if p['cat']==c)
    sc=sum(len(p['scleras']) for p in allp if p['cat']==c)
    print(f'  {c:12s}: {n:3d} patients | face={im} | sclera={sc}')

# ── Stage 1: Binary ──────────────────────────────────────────
print('\n[2] BINARY SCREENING')
bm={'normal':0,'mild':1,'moderate':1,'severe':1}
bf=lambda p: bm[p['cat']]
tr,va=split(allp,bf)
trd=DS(tr,'faces',bm,tr_tf); vad=DS(va,'faces',bm,ev_tf)
trl=DataLoader(trd,BS,shuffle=True,num_workers=0)
vrl=DataLoader(vad,BS,shuffle=False,num_workers=0)
print(f'  Train:{len(tr)} Val:{len(va)} imgs:{len(trd)}/{len(vad)}')
bres={}
for bb in ['convnext_tiny','vit_tiny_patch16_224','efficientnet_b0']:
    print(f'\n  >> {bb}')
    m=timm.create_model(bb,pretrained=True,num_classes=2).to(DEV)
    bres[bb]=train(m,trl,vrl,2,f'Bin-{bb[:10]}',f'v3_binary_{bb}')

# ── Stage 2: Ternary (4 backbones) ──────────────────────────
print('\n[3] TERNARY GRADING')
jp=[p for p in allp if p['cat'] in ('mild','moderate','severe')]
tm={'mild':0,'moderate':1,'severe':2}
tf_=lambda p: tm[p['cat']]
tr,va=split(jp,tf_)
trd=DS(tr,'faces',tm,tr_tf); vad=DS(va,'faces',tm,ev_tf)
trl=DataLoader(trd,BS,shuffle=True,num_workers=0)
vrl=DataLoader(vad,BS,shuffle=False,num_workers=0)
print(f'  Train:{len(tr)} Val:{len(va)} imgs:{len(trd)}/{len(vad)}')
tres={}; tpreds={}
for bb in ['convnext_tiny','vit_tiny_patch16_224','efficientnet_b0','swin_tiny_patch4_window7_224']:
    print(f'\n  >> {bb}')
    m=timm.create_model(bb,pretrained=True,num_classes=3).to(DEV)
    tres[bb]=train(m,trl,vrl,3,f'Ter-{bb[:10]}',f'v3_ternary_{bb}')
    # Get val preds
    m=timm.create_model(bb,pretrained=True,num_classes=3).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL,f'v3_ternary_{bb}.pt'),weights_only=True))
    m.eval()
    ps=[]
    with torch.no_grad():
        for im,_,_ in vrl: ps.extend(F.softmax(m(im.to(DEV)),1).cpu().numpy())
    tpreds[bb]=np.array(ps)
    del m; torch.cuda.empty_cache()

# ── YellowFeatures (sclera) ──────────────────────────────────
print('\n[4] YELLOWFEATURES (sclera)')
jsc=[p for p in jp if p.get('scleras')]
if jsc:
    tr,va=split(jsc,tf_)
    trd=DS(tr,'scleras',tm,tr_tf); vad=DS(va,'scleras',tm,ev_tf)
    trl=DataLoader(trd,BS,shuffle=True,num_workers=0)
    vrl=DataLoader(vad,BS,shuffle=False,num_workers=0)
    print(f'  Sclera: train={len(trd)} val={len(vad)}')
    m=YellowFeatures(3).to(DEV)
    tres['yellowfeatures']=train(m,trl,vrl,3,'YF','v3_ternary_yf')
    m=YellowFeatures(3).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL,'v3_ternary_yf.pt'),weights_only=True))
    m.eval()
    ps=[]
    vls_pids=[]
    with torch.no_grad():
        for im,la,pi in vrl:
            ps.extend(F.softmax(m(im.to(DEV)),1).cpu().numpy()); vls_pids.extend(pi)
    tpreds['yf']={'probs':np.array(ps),'pids':vls_pids}
    del m; torch.cuda.empty_cache()

# ── DynamicFusion Ensemble ───────────────────────────────────
print('\n[5] DYNAMICFUSION ENSEMBLE')
# Use face validation as primary set
_,va_f=split(jp,tf_)
vad_f=DS(va_f,'faces',tm,ev_tf)
vrl_f=DataLoader(vad_f,BS,shuffle=False,num_workers=0)
face_pids=[]
for im,la,pi in vrl_f: face_pids.extend(pi)

# Get YF predictions aligned to face validation patients
yf_probs_by_pid = {}
if 'yf' in tpreds:
    for i, pi in enumerate(tpreds['yf']['pids'][:len(tpreds['yf']['probs'])]):
        yf_probs_by_pid[pi] = tpreds['yf']['probs'][i]

# Build ensemble per patient
pp={}
for bb,probs in tpreds.items():
    if bb=='yf': continue
    for i,pi in enumerate(face_pids[:len(probs)]):
        pp.setdefault(pi,[]).append(probs[i])
        # Add YF prediction if available for this patient
        if pi in yf_probs_by_pid:
            pp[pi].append(yf_probs_by_pid[pi])

epis=list(pp.keys())
eprob=np.array([np.mean(pp[p],0) for p in epis])
plab={p['id']:tm[p['cat']] for p in va_f}
true=np.array([plab[p] for p in epis])
ple=eprob.argmax(1)
try: ea=roc_auc_score(true,eprob,multi_class='ovr')
except: ea=0
eac=accuracy_score(true,ple)
ef1=f1_score(true,ple,average='macro')
ecm=confusion_matrix(true,ple,labels=[0,1,2])
print(f'  Ensemble AUC: {ea:.4f}  Acc: {eac:.4f}  F1: {ef1:.4f}')
print(f'  CM: {ecm.tolist()}')

# ── Summary ──────────────────────────────────────────────────
print('\n'+'='*60)
print('  v3 COMPLETE RESULTS')
print('='*60)
print('\n  Binary:')
for n,a in bres.items(): print(f'    {n:35s}: {a:.4f}')
print('\n  Ternary:')
for n,a in tres.items(): print(f'    {n:35s}: {a:.4f}')
print(f'\n  BilinGuard Ensemble:                    AUC={ea:.4f}')

with open(os.path.join(RES,'v3_results.json'),'w') as f:
    json.dump({'binary':bres,'ternary':tres,'ensemble':ea,'ens_acc':eac,'ens_f1':ef1},f,indent=2,default=str)
