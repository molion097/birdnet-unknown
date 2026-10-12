"""Sensitivity of the routing gain to the group-level synonym uncertainty (Section 3).

A loose, unreviewed group-level synonym check (taxonomy_crosswalk.py: a species row plus its ssp rows, any
of the IOC/AviList/Clements/HBW/H&M list columns) flags up to about 30/131 (Indonesia), 11/53 (Philippines)
and 8/39 (Malaysia) species "unknown to BirdNET" that may be known to BirdNET under a different name. The
routing result (Table 1) depends on which species count as unknown, so this script reclassifies every such
species as known and reruns the comparison. This is the worst case for the routing gain, not a claim that
the reclassification is correct.

Method: rebuild each region's per-recording arrays (species, recordist, top1, s1, known, truth) as
table1_birdnet_ID_PH.py and table1_birdnet_MY.py do. No model is refitted: the extension
classifier outputs (eo, ptop) are read from data/model_outputs/extension_cache_ID.npz, extension_cache_PH.npz
and extension_cache_MY.npz. Only the known flag and BirdNET's correctness (bn_ok) change, using the same
group-level alias construction as taxonomy_crosswalk.py (same IOC workbook). For recordings of a
reclassified species, known_v2 is True and bn_ok_v2 is True only if BirdNET's own top-1 prediction is one
of that species' linked alias names. Recordings of other species keep their original values (checked
against the cached s1 and bn_ok for PH and MY).

The 25-halving corrected paired-t comparison of routing against BirdNET-only (same code and seeds as
experiments 21 and 22) is then run on the original and the reclassified accounting, for
r in (0.05, 0.10, 0.20, 0.30), and both are printed side by side per region.

Reads data/bench_files_{ID,PH,MY}_cap6.csv, the matching prediction files and caches in data/model_outputs/,
data/model_outputs/bench_ID6_labels.json and the IOC 15.2 cross-reference workbook.
Usage: python check_name_reconciliation.py
"""
import csv, json, sys, collections
import numpy as np
from scipy import stats
import openpyxl

D = __import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/../data/'
sci = lambda l: l.split('_')[0]

# Alias groups (galias), built as in taxonomy_crosswalk.py.
wb = openpyxl.load_workbook(__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/../external/IOC_15.2_vs_other_lists.xlsx', read_only=True)
ws = wb.worksheets[0]
allrows = list(ws.iter_rows(min_row=2, values_only=True))
rows_sp = [r for r in allrows if r[2] == 'species']
groups = []
cur = None
for r in allrows:
    if r[2] == 'species':
        cur = [r]; groups.append(cur)
    elif cur is not None and r[2] == 'ssp':
        cur.append(r)
galias = collections.defaultdict(set)
for g in groups:
    names = {' '.join(str(x).split()[:2]) for r in g for x in r[1:] if isinstance(x, str) and len(str(x).split()) >= 2}
    ioc = g[0][1]
    for n in names:
        galias[n] |= names | {ioc}

bl = {s.split('_')[0] for s in json.load(open(D + 'model_outputs/bench_ID6_labels.json'))}

CFG = {
    'ID': dict(csv='bench_files_ID_cap6.csv', pred='bench_ID6_predictions.json', cache='extension_cache_ID.npz', has_bnok=False),
    'PH': dict(csv='bench_files_PH_cap6.csv', pred='bench_PH6emb_predictions.json', cache='extension_cache_PH.npz', has_bnok=True),
    'MY': dict(csv='bench_files_MY_cap6.csv', pred='bench_MY6emb_predictions.json', cache='extension_cache_MY.npz', has_bnok=True),
}

def choose(wv, cv, wt, ct, r):
    f = np.where(wv <= r + 1e-12)[0]
    if not len(f):
        return None
    i = f[np.argmax(cv[f])]
    return ct[i]

def hybrid_curve(s1, bn_ok, pm, eo, m, ths):
    ab = s1[m][None, :] >= ths[:, None]
    ae = pm[m][None, :] >= ths[:, None]
    okb = bn_ok[m]; oke = eo[m]
    W = np.zeros((101, 101)); C = np.zeros((101, 101))
    for i in range(101):
        ub = ab[i]; ue = (~ub)[None, :] & ae
        W[i] = (ub & ~okb).mean() + (ue & ~oke[None, :]).mean(1)
        C[i] = (ub & okb).mean() + (ue & oke[None, :]).mean(1)
    return W.ravel(), C.ravel()

def base_curve(s1, bn_ok, m, ths):
    a = s1[m][None, :] >= ths[:, None]; ok = bn_ok[m][None, :]
    return (a & ~ok).mean(1), (a & ok).mean(1)

def run_region(region):
    cfg = CFG[region]
    rows = list(csv.DictReader(open(D + cfg['csv'], encoding='utf8')))
    P = {x['file']: x['top'] for x in json.load(open(D + 'model_outputs/' + cfg['pred']))}
    rows = [r for r in rows if r['file'] in P]
    cache = np.load(D + 'model_outputs/' + cfg['cache'])
    assert len(rows) == len(cache['eo']), f'{region}: row count {len(rows)} != cache length {len(cache["eo"])}'
    eo = cache['eo']; pm = cache['ptop']

    sp = np.array([r['species'] for r in rows])
    rec = np.array([r['recordist'] or 'NA' for r in rows])
    known = np.array([bool(r['birdnet_label']) for r in rows])
    truth = np.array([r['birdnet_label'] for r in rows])
    top1 = np.array([sci(max(P[r['file']], key=lambda t: t[1])[0]) for r in rows])
    s1 = np.array([max(s for _, s in P[r['file']]) for r in rows])
    bn_ok = known & (top1 == truth)

    if cfg['has_bnok']:
        # sanity check: the recomputed arrays must match the cached ones exactly
        mism_s1 = np.abs(s1 - cache['s1']).max()
        mism_bnok = (bn_ok != cache['bn_ok']).sum()
        print(f'  [sanity check {region}] max|s1 diff|={mism_s1:.6f}, bn_ok mismatches={mism_bnok}/{len(rows)}')
        assert mism_s1 < 1e-5 and mism_bnok == 0, f'{region}: recomputed arrays do not match cache -- alignment bug, stopping'

    # species currently flagged unknown to BirdNET
    unk_species = sorted({s for s, k in zip(sp, known) if not k})
    # group-level (upper-bound) alias links, as in the "GROUP-level links" of taxonomy_crosswalk.py
    glinked = {}
    for s in unk_species:
        if s in galias:
            aliases = sorted(a for a in galias[s] if a in bl and a != s)
            if aliases:
                glinked[s] = aliases

    n_unk = len(unk_species)
    print(f'  {region}: {n_unk} species unknown to BirdNET, {len(glinked)} have a group-level (upper-bound) alias link')
    for s, a in glinked.items():
        print(f'     {s} -> {a}')

    # reclassify: known_v2 = True for recordings of a linked species. bn_ok_v2 = True if BirdNET's own
    # top1 prediction is one of that species' linked aliases (BirdNET already names it, under a synonym),
    # else False (still wrong, but no longer counted as unfixable by any threshold)
    reclass_mask = np.array([s in glinked for s in sp])
    bn_ok_v2 = bn_ok.copy()
    for i in np.where(reclass_mask)[0]:
        bn_ok_v2[i] = top1[i] in glinked[sp[i]]
    known_v2 = known | reclass_mask

    print(f'  {region}: reclassified {reclass_mask.sum()} recordings ({reclass_mask.sum()/len(rows)*100:.1f}% of all); '
          f'of those, BirdNET\'s own top1 already matched an alias for {bn_ok_v2[reclass_mask].sum()} ({bn_ok_v2[reclass_mask].mean()*100:.1f}%)')

    ths = np.linspace(0, 1, 101)
    ur = np.array(sorted(set(rec)))
    results = {}
    for label, bnk in (('original', bn_ok), ('reclassified', bn_ok_v2)):
        res = collections.defaultdict(dict)
        for run in range(25):
            rng = np.random.default_rng(run)
            half = set(rng.permutation(ur)[:len(ur) // 2])
            inv = np.array([r in half for r in rec])
            for r in (0.05, 0.10, 0.20, 0.30):
                wv, cv = base_curve(s1, bnk, inv, ths); wt, ct = base_curve(s1, bnk, ~inv, ths)
                c_bn = choose(wv, cv, wt, ct, r)
                wv, cv = hybrid_curve(s1, bnk, pm, eo, inv, ths); wt, ct = hybrid_curve(s1, bnk, pm, eo, ~inv, ths)
                c_hy = choose(wv, cv, wt, ct, r)
                if c_bn is not None and c_hy is not None:
                    res[r][run] = (c_bn, c_hy)
        results[label] = res
    return results, n_unk, len(glinked), reclass_mask.sum(), len(rows)

print('=' * 100)
print('Sensitivity check: routing-vs-BirdNET gain under group-level (upper-bound) synonym reclassification')
print('=' * 100)
for region in ('ID', 'PH', 'MY'):
    print(f'\n--- {region} ---')
    results, n_unk, n_linked, n_reclass_rec, n_rows = run_region(region)
    for r in (0.05, 0.10, 0.20, 0.30):
        line = f'  r={r:.2f}  '
        for label in ('original', 'reclassified'):
            v = results[label][r]
            ks = sorted(v)
            d_bn = np.array([v[k][0] for k in ks])
            d_hy = np.array([v[k][1] for k in ks])
            diff = d_hy - d_bn
            J = len(diff); sd = diff.std(ddof=1) * np.sqrt(1 / J + 1); h = stats.t.ppf(.975, J - 1) * sd
            line += f'{label}: BirdNET={d_bn.mean():.3f} routing={d_hy.mean():.3f} gain={diff.mean():+.4f} [{diff.mean()-h:+.4f},{diff.mean()+h:+.4f}] (n={J})   '
        print(line)
