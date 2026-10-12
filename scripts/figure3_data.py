"""Data for the risk-coverage figure (Figure 3).

For Indonesia and Malaysia, recomputes BirdNET's top score (s1) and top-1 correctness (bn_ok) per recording from
the stored predictions, without retraining, and combines them with the cached extension-classifier outputs
(ptop, eo) in data/model_outputs/extension_cache_ID.npz and extension_cache_MY.npz. The embedding files are read
only to fix the recording set and order. For the Philippines, table1_birdnet_ID_PH.py already stores s1 and bn_ok in
extension_cache_PH.npz, so they are taken from there.

Writes data/model_outputs/fig_curve_ID.npz, fig_curve_PH.npz and fig_curve_MY.npz.
"""
import csv,json,numpy as np
D=__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/../data/'
sci=lambda l:l.split('_')[0]

def region_s1_bnok(csvname,predname,embname):
    rows=list(csv.DictReader(open(D+csvname,encoding='utf8')))
    z=np.load(D+'model_outputs/'+embname); idx={f:i for i,f in enumerate(z['files'])}
    P={x['file']:x['top'] for x in json.load(open(D+'model_outputs/'+predname))}
    rows=[r for r in rows if r['file'] in idx and r['file'] in P]
    known=np.array([bool(r['birdnet_label']) for r in rows]); truth=np.array([r['birdnet_label'] for r in rows])
    top1=np.array([sci(max(P[r['file']],key=lambda t:t[1])[0]) for r in rows]); s1=np.array([max(s for _,s in P[r['file']]) for r in rows])
    bn_ok=known&(top1==truth)
    return s1,bn_ok,len(rows)

s1,bn_ok,n=region_s1_bnok('bench_files_ID_cap6.csv','bench_ID6_predictions.json','bench_ID6_embeddings.npz')
print('ID',n,'s1/bn_ok computed')
cache=np.load(D+'model_outputs/extension_cache_ID.npz')
assert len(cache['ptop'])==n, (len(cache['ptop']),n)
np.savez(D+'model_outputs/fig_curve_ID.npz',s1=s1,bn_ok=bn_ok,ptop=cache['ptop'],eo=cache['eo'])
print('saved fig_curve_ID.npz')

s1,bn_ok,n=region_s1_bnok('bench_files_MY_cap6.csv','bench_MY6emb_predictions.json','bench_MY6emb_embeddings.npz')
print('MY',n,'s1/bn_ok computed')
cache=np.load(D+'model_outputs/extension_cache_MY.npz')
assert len(cache['ptop'])==n, (len(cache['ptop']),n)
np.savez(D+'model_outputs/fig_curve_MY.npz',s1=s1,bn_ok=bn_ok,ptop=cache['ptop'],eo=cache['eo'])
print('saved fig_curve_MY.npz')

cache=np.load(D+'model_outputs/extension_cache_PH.npz')   # table1_birdnet_ID_PH.py stores s1 and bn_ok with the classifier outputs
np.savez(D+'model_outputs/fig_curve_PH.npz',s1=cache['s1'],bn_ok=cache['bn_ok'],ptop=cache['ptop'],eo=cache['eo'])
print('saved fig_curve_PH.npz')
