"""Table 2: effect of the BirdNET range filter (threshold 0.03, year-round lists) in each region.

Uses the per-recording species lists in data/model_outputs/bench_<REG>6_geo_lists.json, which the BirdNET V2.4 geo
model produced from the recording coordinates (species with occurrence probability >= 0.03). The released tables
do not contain exact coordinates, so the lists are provided instead. Recordings without a list (no coordinates)
and recordings missing from the predictions file are excluded.

For each region the script prints the recall of the true species in the list (BirdNET-known recordings), known
top-1 accuracy without and with the filter (exact McNemar test), and correct_all at validation wrong_all <= 0.10
without and with the filter, with the filtered-minus-unfiltered difference and its corrected paired-t 95% interval
(species-disjoint 5 folds x 5 seeds, variance factor 1/J + 1).

Usage: python table2_range_filter.py [ID] [TW] [PH] [MY]   (default: all four)
"""
import csv, json, sys
import numpy as np
from scipy import stats
from scipy.stats import binomtest

D = __import__('os').path.dirname(__import__('os').path.abspath(__file__)) + '/../data/'
REGIONS = {'ID': ('Indonesia', 'bench_files_ID_cap6_wav.csv', 'bench_ID6_predictions.json', 'bench_ID6_geo_lists.json'),
           'TW': ('Taiwan', 'bench_files_TW_cap6_wav.csv', 'bench_TW6_predictions.json', 'bench_TW6_geo_lists.json'),
           'PH': ('Philippines', 'bench_files_PH_cap6_wav.csv', 'bench_PH6_predictions.json', 'bench_PH6_geo_lists.json'),
           'MY': ('Malaysia', 'bench_files_MY_cap6_wav.csv', 'bench_MY6_predictions.json', 'bench_MY6_geo_lists.json')}


def sci(label):
    return label.split('_')[0]


def region(tag, csvname, pred, lists):
    rows = list(csv.DictReader(open(D + csvname, encoding='utf8')))
    G = json.load(open(D + 'model_outputs/' + lists))
    Pr = {x['file']: x['top'] for x in json.load(open(D + 'model_outputs/' + pred))}
    n0 = len(rows)
    rows = [r for r in rows if G.get(r['file']) is not None and r['file'] in Pr]
    print(f'{tag}: {n0 - len(rows)} recordings without coordinates or predictions excluded')
    L = [set(G[r['file']]) for r in rows]
    known = np.array([bool(r.get('birdnet_label')) for r in rows])
    truth = [r.get('birdnet_label') or '' for r in rows]
    sp = np.array([r['species'] for r in rows])
    n = len(rows)
    recall = np.mean([truth[i] in L[i] for i in range(n) if known[i]])

    def top1(i, filt):
        for label, score in Pr[rows[i]['file']]:
            name = sci(label)
            if not filt or name in L[i]:
                return name, score
        return '', 0.0

    res = {}
    for filt in (False, True):
        T = [top1(i, filt) for i in range(n)]
        res[filt] = (np.array([t[1] for t in T]), np.array([bool(truth[i]) and T[i][0] == truth[i] for i in range(n)]))
    b = int((res[False][1][known] & ~res[True][1][known]).sum())
    c = int((res[True][1][known] & ~res[False][1][known]).sum())
    p_mcnemar = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0

    ths = np.linspace(0, 1, 201)
    out = {False: {}, True: {}}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        spp = sorted(set(sp))
        rng.shuffle(spp)
        fold_of = {q: i % 5 for i, q in enumerate(spp)}
        fold = np.array([fold_of[q] for q in sp])
        for t in range(5):
            v = (t + 1) % 5
            for filt in (False, True):
                s, ok = res[filt]

                def curve(mask):
                    a = s[mask][None, :] >= ths[:, None]
                    return (a & ~ok[mask][None, :]).mean(axis=1), (a & ok[mask][None, :]).mean(axis=1)

                wv, cv = curve(fold == v)
                wt, ct = curve(fold == t)
                f = np.where(wv <= 0.10 + 1e-12)[0]
                if len(f):
                    out[filt][seed * 5 + t] = ct[f[np.argmax(cv[f])]]
    common = sorted(set(out[True]) & set(out[False]))
    d = np.array([out[True][i] - out[False][i] for i in common])
    J = len(d)
    se = np.sqrt((1 / J + 1) * d.std(ddof=1) ** 2)
    tq = stats.t.ppf(0.975, J - 1)
    print(f'{tag}: {n} recordings with coordinates, recall {recall:.3f}, '
          f'known top-1 {res[False][1][known].mean():.3f} -> {res[True][1][known].mean():.3f} '
          f'(McNemar: correct only unfiltered {b}, only filtered {c}, exact p {p_mcnemar:.2g}); '
          f'correct_all at r=0.10: unfiltered {np.mean([out[False][i] for i in common]):.3f}, '
          f'filtered {np.mean([out[True][i] for i in common]):.3f}, '
          f'change {d.mean():+.3f} [{d.mean() - tq * se:+.3f}, {d.mean() + tq * se:+.3f}]')


for key in (sys.argv[1:] or ['ID', 'TW', 'PH', 'MY']):
    region(*REGIONS[key])
