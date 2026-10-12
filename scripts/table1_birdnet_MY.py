"""Table 1, BirdNET columns for Malaysia.

Same method as table1_birdnet_ID_PH.py: BirdNET
embedding, score profile and range-list bit, LinearSVC, and a temperature-scaled softmax calibrated on a
recordist-disjoint slice within each outer GroupKFold(5)-by-recordist fold. Applied to the Malaysia cap-6
benchmark (1,968 recordings, 328 species). Only 39/328 species (12%) are outside BirdNET's label set,
against 33% in Indonesia and 38% in the Philippines, so this region tests whether the gain holds when few
species are unknown to BirdNET. Evaluation: 25 recordist-disjoint validation/test halvings, correct_all at
wrong_all <= r for BirdNET-only, the stacked classifier alone and the two-threshold hybrid, for r in
(0.05, 0.10, 0.20, 0.30). Corrected-t (Nadeau-Bengio, 1/J+1) 95% CI against BirdNET-only, and of stacked
against hybrid. No resampling.

Writes data/model_outputs/extension_cache_MY.npz (used by check_name_reconciliation.py and figure3_data.py).
Usage: python table1_birdnet_MY.py
"""
import csv,json,numpy as np,collections,os
import scipy.sparse as sp_
from scipy.optimize import minimize_scalar
from scipy import stats
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupKFold

REGION=os.environ.get('REGION','MY')
D=__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/../data/'
CFG={
    'ID': dict(csv='bench_files_ID_cap6.csv', emb='bench_ID6_embeddings.npz', pred='bench_ID6_predictions.json', geo='bench_ID6_geo_lists.json'),
    'PH': dict(csv='bench_files_PH_cap6.csv', emb='bench_PH6emb_embeddings.npz', pred='bench_PH6emb_predictions.json', geo='bench_PH6_geo_lists.json'),
    'MY': dict(csv='bench_files_MY_cap6.csv', emb='bench_MY6emb_embeddings.npz', pred='bench_MY6emb_predictions.json', geo='bench_MY6_geo_lists.json'),
}[REGION]

rows=list(csv.DictReader(open(D+CFG['csv'],encoding='utf8')))
z=np.load(D+'model_outputs/'+CFG['emb']); idx={f:i for i,f in enumerate(z['files'])}
P={x['file']:x['top'] for x in json.load(open(D+'model_outputs/'+CFG['pred']))}
G=json.load(open(D+'model_outputs/'+CFG['geo']))
rows=[r for r in rows if r['file'] in idx and r['file'] in P]
print(REGION,'recordings',len(rows),'species',len({r['species'] for r in rows}),flush=True)
XB=z['emb'][[idx[r['file']] for r in rows]].astype(np.float32); XB=XB/(np.linalg.norm(XB,axis=1,keepdims=True)+1e-9)
lab={l:i for i,l in enumerate(sorted({l for r in rows for l,_ in P[r['file']]}))}
data=[];ri=[];ci=[]
for i,r in enumerate(rows):
    for l,s in P[r['file']]: ri.append(i); ci.append(lab[l]); data.append(float(s))
Sp=normalize(sp_.csr_matrix((data,(ri,ci)),shape=(len(rows),len(lab)),dtype=np.float32))
sci=lambda l:l.split('_')[0]
geo=np.full(len(rows),0.5,np.float32)
for i,r in enumerate(rows):
    g=G.get(r['file'])
    if g is not None:
        t1=sci(max(P[r['file']],key=lambda t:t[1])[0]); geo[i]=1.0 if t1 in set(g) else 0.0
F=sp_.hstack([sp_.csr_matrix(XB),Sp,sp_.csr_matrix(geo.reshape(-1,1))]).tocsr()
sp=np.array([r['species'] for r in rows]); rec=np.array([r['recordist'] or 'NA' for r in rows]); known=np.array([bool(r['birdnet_label']) for r in rows]); truth=np.array([r['birdnet_label'] for r in rows])
top1=np.array([sci(max(P[r['file']],key=lambda t:t[1])[0]) for r in rows]); s1=np.array([max(s for _,s in P[r['file']]) for r in rows]); bn_ok=known&(top1==truth)
classes=sorted(set(sp)); cidx={c:i for i,c in enumerate(classes)}; y=np.array([cidx[s] for s in sp])
prob=np.zeros((len(rows),len(classes)),np.float32)
def softmax(zz):
    zz=zz-zz.max(1,keepdims=True); e=np.exp(zz); return e/e.sum(1,keepdims=True)
gk=GroupKFold(n_splits=min(5,len(set(rec))))
for tr,te in gk.split(F,y,groups=rec):
    rtr=rec[tr]; urtr=np.array(sorted(set(rtr))); rng=np.random.default_rng(0); cal_r=set(rng.choice(urtr,max(1,len(urtr)//5),replace=False))
    cal=np.array([r in cal_r for r in rtr]); fit_=~cal
    clf=LinearSVC(C=1.0,max_iter=5000,random_state=0).fit(F[tr][fit_],y[tr][fit_])
    dz=clf.decision_function(F[tr][cal])
    if dz.ndim==1: dz=np.c_[-dz,dz]
    yz=y[tr][cal]
    def nll(T):
        p=softmax(dz/np.exp(T)); ii=np.searchsorted(clf.classes_,yz); ii=np.clip(ii,0,p.shape[1]-1)
        pp=p[np.arange(len(yz)),ii]; return -np.log(np.clip(pp,1e-9,1)).mean()
    Topt=minimize_scalar(nll,bounds=(-3,3),method='bounded').x
    dte=clf.decision_function(F[te])
    if dte.ndim==1: dte=np.c_[-dte,dte]
    pr=softmax(dte/np.exp(Topt)); prob[np.ix_(te,clf.classes_)]=pr
pe=prob.argmax(1); pm=prob.max(1); eo=pe==y
order=np.argsort(-prob,axis=1); pmargin=prob[np.arange(len(rows)),order[:,0]]-prob[np.arange(len(rows)),order[:,1]]
np.savez(D+'model_outputs/extension_cache_MY.npz',ptop=pm,pmargin=pmargin,eo=eo,s1=s1,bn_ok=bn_ok)
print('stacked classifier top-1 overall %.3f known %.3f unknown %.3f (BirdNET overall %.3f known %.3f)'%(eo.mean(),eo[known].mean() if known.any() else float('nan'),eo[~known].mean() if (~known).any() else float('nan'),bn_ok.mean(),bn_ok[known].mean()),flush=True)
ths=np.linspace(0,1,101); ur=np.array(sorted(set(rec)))
def base_curve(m): a=s1[m][None,:]>=ths[:,None]; ok=bn_ok[m][None,:]; return (a&~ok).mean(1),(a&ok).mean(1)
def stack_curve(m): a=pm[m][None,:]>=ths[:,None]; ok=eo[m][None,:]; return (a&~ok).mean(1),(a&ok).mean(1)
def hybrid_curve(m):
    ab=s1[m][None,:]>=ths[:,None]; ae=pm[m][None,:]>=ths[:,None]; okb=bn_ok[m]; oke=eo[m]; W=np.zeros((101,101)); C=np.zeros((101,101))
    for i in range(101):
        ub=ab[i]; ue=(~ub)[None,:]&ae; W[i]=(ub&~okb).mean()+(ue&~oke[None,:]).mean(1); C[i]=(ub&okb).mean()+(ue&oke[None,:]).mean(1)
    return W.ravel(),C.ravel()
def choose(wv,cv,wt,ct,r):
    f=np.where(wv<=r+1e-12)[0]
    if not len(f): return None
    i=f[np.argmax(cv[f])]; return ct[i]
res=collections.defaultdict(dict)
for run in range(25):
    rng=np.random.default_rng(run); half=set(rng.permutation(ur)[:len(ur)//2]); inv=np.array([r in half for r in rec])
    for r in (0.05,0.10,0.20,0.30):
        for nm,fn in (('birdnet',base_curve),('hybrid',hybrid_curve),('stacked',stack_curve)):
            wv,cv=fn(inv); wt,ct=fn(~inv); c=choose(wv,cv,wt,ct,r)
            if c is not None: res[(nm,r)][run]=c
for r in (0.05,0.10,0.20,0.30):
    for nm in ('birdnet','hybrid','stacked'):
        v=res[(nm,r)]; print(f'r={r} {nm:8s} runs={len(v)} correct_all {np.mean(list(v.values())):.3f}',flush=True)
    for a in ('hybrid','stacked'):
        ks=sorted(set(res[(a,r)])&set(res[('birdnet',r)])); d=np.array([res[(a,r)][k]-res[('birdnet',r)][k] for k in ks]); J=len(d); sd=d.std(ddof=1)*np.sqrt(1/J+1); h=stats.t.ppf(.975,J-1)*sd
        print(f'   {a}-birdnet: {d.mean():+.4f} [{d.mean()-h:+.4f},{d.mean()+h:+.4f}] n={J}',flush=True)
    ks=sorted(set(res[('stacked',r)])&set(res[('hybrid',r)])); d=np.array([res[('stacked',r)][k]-res[('hybrid',r)][k] for k in ks]); J=len(d); sd=d.std(ddof=1)*np.sqrt(1/J+1); h=stats.t.ppf(.975,J-1)*sd
    print(f'   stacked-hybrid: {d.mean():+.4f} [{d.mean()-h:+.4f},{d.mean()+h:+.4f}] n={J}',flush=True)
