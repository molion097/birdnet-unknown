"""Cosine-centroid baseline following Jana et al. (2026, Ecology and Evolution 16:e74335; preprint arXiv:2504.16276, Sections 2.6 and 3.3).

Their method: PCA keeping 95% of the variance, L2-normalise the reduced embeddings, take the median
embedding of each species as its centroid, compute the cosine similarity between a query and every
centroid, and apply a softmax to get a top-1 prediction. Nothing is trained. This differs from the
nearest-centroid ablation in the paper (sklearn NearestCentroid: Euclidean distance to the mean of the
raw embeddings, no PCA) in three ways: PCA, median instead of mean, cosine instead of Euclidean.

Calibration: Jana et al. use a plain softmax over the similarities. Their task has only a few
classes, ours has 397, 138 and 328. With cosine similarities clustered near 0 (sd about 0.10),
a plain softmax over hundreds of classes is near-uniform (maximum probability about 0.004), so no
threshold can select a confident subset and the routing rule cannot work. We therefore add the same
temperature calibration as for our SVM (NLL minimised on a recordist-disjoint calibration slice of
each training fold). This step is our addition, not part of their method. The comparison is then
between the two classifiers, both calibrated the same way.

Evaluation is the same as for the SVM extension classifier: recordist-disjoint GroupKFold(5)
cross-fitting and 25 validation/test halvings for the routing rule. PCA and centroids are fit on the
training part of each fold only. Reads the benchmark CSVs in data/ and the BirdNET embeddings and
predictions in data/model_outputs/; prints top-1 accuracy and routing results to stdout. CPU only.
"""
import csv, json, collections
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

D = __import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/../data/'
sci = lambda l: l.split('_')[0]

CFG = {
    'ID': dict(csv='bench_files_ID_cap6.csv', emb='bench_ID6_embeddings.npz', pred='bench_ID6_predictions.json'),
    'PH': dict(csv='bench_files_PH_cap6.csv', emb='bench_PH6emb_embeddings.npz', pred='bench_PH6emb_predictions.json'),
    'MY': dict(csv='bench_files_MY_cap6.csv', emb='bench_MY6emb_embeddings.npz', pred='bench_MY6emb_predictions.json'),
}
# linear-SVM extension classifier results (Table 1, Section 5.2), printed for comparison
SVM_TOP1 = {'ID': (0.571, 0.694, 0.322), 'PH': (0.611, 0.745, 0.396), 'MY': (None, None, None)}  # overall/known/unknown; no top-1 for MY is given in the text

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def run_region(region):
    cfg = CFG[region]
    rows = list(csv.DictReader(open(D + cfg['csv'], encoding='utf8')))
    z = np.load(D + 'model_outputs/' + cfg['emb']); idx = {f: i for i, f in enumerate(z['files'])}
    P = {x['file']: x['top'] for x in json.load(open(D + 'model_outputs/' + cfg['pred']))}
    rows = [r for r in rows if r['file'] in idx and r['file'] in P]
    X = z['emb'][[idx[r['file']] for r in rows]].astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    sp = np.array([r['species'] for r in rows])
    rec = np.array([r['recordist'] or 'NA' for r in rows])
    known = np.array([bool(r['birdnet_label']) for r in rows])
    truth = np.array([r['birdnet_label'] for r in rows])
    top1bn = np.array([sci(max(P[r['file']], key=lambda t: t[1])[0]) for r in rows])
    s1 = np.array([max(s for _, s in P[r['file']]) for r in rows])
    bn_ok = known & (top1bn == truth)
    classes = sorted(set(sp)); cidx = {c: i for i, c in enumerate(classes)}; y = np.array([cidx[c] for c in sp])

    pm = np.zeros(len(rows)); eo = np.zeros(len(rows), dtype=bool)
    for tr, te in GroupKFold(5).split(X, y, groups=rec):
        pca = PCA(n_components=0.95, svd_solver='full').fit(X[tr])
        Xtr = pca.transform(X[tr]); Xte = pca.transform(X[te])
        Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-9)
        Xte = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
        ytr = y[tr]

        # recordist-disjoint calibration slice within this training fold, built the same way as for
        # the SVM's temperature scaling
        rtr = rec[tr]; urtr = np.array(sorted(set(rtr)))
        cal_rng = np.random.default_rng(0); cal_r = set(cal_rng.choice(urtr, max(1, len(urtr) // 5), replace=False))
        cal = np.array([r in cal_r for r in rtr]); fit_ = ~cal

        # per-class median centroid (as in Jana et al.) from the fit portion, re-normalised
        present = sorted(set(ytr[fit_]))
        pidx = {c: i for i, c in enumerate(present)}
        cent = np.zeros((len(present), Xtr.shape[1]))
        for i, c in enumerate(present):
            m = np.median(Xtr[fit_][ytr[fit_] == c], axis=0)
            cent[i] = m / (np.linalg.norm(m) + 1e-9)

        sim_cal = Xtr[cal] @ cent.T  # cosine similarity, both sides unit-norm
        ycal = ytr[cal]
        cal_has_class = np.array([c in pidx for c in ycal])  # keep calibration recordings whose species has a centroid
        def nll(T):
            p = softmax(sim_cal[cal_has_class] / np.exp(T))
            ii = np.array([pidx[c] for c in ycal[cal_has_class]])
            pp = p[np.arange(len(ii)), ii]
            return -np.log(np.clip(pp, 1e-9, 1)).mean()
        Topt = minimize_scalar(nll, bounds=(-3, 5), method='bounded').x if cal_has_class.sum() else 0.0

        sim_te = Xte @ cent.T
        prob_present = softmax(sim_te / np.exp(Topt))
        pred_local = prob_present.argmax(1)
        pm[te] = prob_present.max(1)
        eo[te] = np.array([present[pred_local[i]] for i in range(len(te))]) == y[te]

    n_known = known.sum(); n_unknown = (~known).sum()
    top1_overall = eo.mean(); top1_known = eo[known].mean() if n_known else float('nan'); top1_unknown = eo[~known].mean() if n_unknown else float('nan')
    print(f'{region}: cosine-centroid (Jana-et-al.-style) top-1 overall {top1_overall:.3f} known {top1_known:.3f} unknown {top1_unknown:.3f}  '
          f'(SVM extension classifier for comparison: {SVM_TOP1[region]})')

    ths = np.linspace(0, 1, 101); ur = np.array(sorted(set(rec)))
    def base_curve(m): a = s1[m][None, :] >= ths[:, None]; ok = bn_ok[m][None, :]; return (a & ~ok).mean(1), (a & ok).mean(1)
    def stack_curve(m): a = pm[m][None, :] >= ths[:, None]; ok = eo[m][None, :]; return (a & ~ok).mean(1), (a & ok).mean(1)
    def hybrid_curve(m):
        ab = s1[m][None, :] >= ths[:, None]; ae = pm[m][None, :] >= ths[:, None]; okb = bn_ok[m]; oke = eo[m]
        W = np.zeros((101, 101)); C = np.zeros((101, 101))
        for i in range(101):
            ub = ab[i]; ue = (~ub)[None, :] & ae
            W[i] = (ub & ~okb).mean() + (ue & ~oke[None, :]).mean(1); C[i] = (ub & okb).mean() + (ue & oke[None, :]).mean(1)
        return W.ravel(), C.ravel()
    def choose(wv, cv, wt, ct, r):
        f = np.where(wv <= r + 1e-12)[0]
        if not len(f): return None
        i = f[np.argmax(cv[f])]; return ct[i]
    res = collections.defaultdict(dict)
    for run in range(25):
        rng = np.random.default_rng(run)
        half = set(rng.permutation(ur)[:len(ur) // 2]); inv = np.array([rr in half for rr in rec])
        for r in (0.05, 0.10, 0.20, 0.30):
            for nm, fn in (('birdnet', base_curve), ('hybrid', hybrid_curve), ('cosine', stack_curve)):
                wv, cv = fn(inv); wt, ct = fn(~inv); c = choose(wv, cv, wt, ct, r)
                if c is not None: res[(nm, r)][run] = c
    for r in (0.05, 0.10, 0.20, 0.30):
        line = f'  r={r:.2f}  '
        for nm in ('birdnet', 'hybrid', 'cosine'):
            v = res[(nm, r)]; line += f'{nm}={np.mean(list(v.values())):.3f}  '
        ks = sorted(set(res[('cosine', r)]) & set(res[('birdnet', r)]))
        d = np.array([res[('cosine', r)][k] - res[('birdnet', r)][k] for k in ks])
        J = len(d); sd = d.std(ddof=1) * np.sqrt(1 / J + 1); h = stats.t.ppf(.975, J - 1) * sd
        line += f'cosine-birdnet={d.mean():+.4f} [{d.mean()-h:+.4f},{d.mean()+h:+.4f}] n={J}'
        print(line)

print('Cosine-similarity-to-median-centroid baseline (jana2025toothbilled\'s actual method), r=0.05/0.10/0.20/0.30')
print('=' * 100)
for region in ('ID', 'PH', 'MY'):
    run_region(region)
    print()
