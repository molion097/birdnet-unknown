"""Plot correct_all against wrong_all over the full threshold range (risk-coverage style figure).

Three curves per region (Indonesia, Philippines, Malaysia): BirdNET alone, the extension classifier alone, and the
two-threshold routing rule. Input arrays are data/model_outputs/fig_curve_<REG>.npz, built by figure3_data.py from
the cross-fitted extension-classifier outputs of the Table 1 scripts. Every prediction in these arrays is already
out-of-sample (recordist-disjoint folds), so the curves use the whole dataset without a validation/test split.
The Table 1 operating points, which come from 25 random validation/test halvings, are overlaid as markers for reference.
They need not lie exactly on the full-sample curves. Dashed curves use Perch 2.0 as the backbone
(data/model_outputs/perch2_cache_<REG>.npz from table1_perch2.py); their BirdNET-score
thresholds are quantiles of the score, as in that script. Output: fig_hybrid_curves.pdf in the release root."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/../data/model_outputs/'

def curves(s1,bn_ok,pm,eo,ths_base=None):
    ths=np.linspace(0,1,101)
    if ths_base is None: ths_base=ths
    def base(): a=s1[None,:]>=ths_base[:,None]; ok=bn_ok[None,:]; return (a&~ok).mean(1),(a&ok).mean(1)
    def stack(): a=pm[None,:]>=ths[:,None]; ok=eo[None,:]; return (a&~ok).mean(1),(a&ok).mean(1)
    def hybrid():
        ab=s1[None,:]>=ths_base[:,None]; ae=pm[None,:]>=ths[:,None]; okb=bn_ok; oke=eo
        W=np.zeros((101,101)); C=np.zeros((101,101))
        for i in range(101):
            ub=ab[i]; ue=(~ub)[None,:]&ae
            W[i]=(ub&~okb).mean()+(ue&~oke[None,:]).mean(1); C[i]=(ub&okb).mean()+(ue&oke[None,:]).mean(1)
        # Pareto frontier over the 101x101 grid: for each wrong_all bucket, best correct_all
        w=W.ravel(); c=C.ravel(); order=np.argsort(w); w=w[order]; c=c[order]
        bestc=np.maximum.accumulate(c)
        return w,bestc
    wb,cb=base(); ws,cs=stack(); wh,ch=hybrid()
    return (wb,cb),(ws,cs),(wh,ch)

# Table 1 operating points (from the manuscript, 25-halving protocol; overlaid for reference only)
table1={
 'ID': {'r':[0.05,0.10,0.20,0.30],'birdnet':[0.282,0.354,0.406,0.431],'routing':[0.347,0.431,0.517,0.560],'ext':[0.320,0.412,0.499,0.547]},
 'PH': {'r':[0.05,0.10,0.20,0.30],'birdnet':[0.219,0.266,0.343,0.372],'routing':[0.350,0.446,0.552,0.600],'ext':[0.344,0.438,0.536,0.588]},
 'MY': {'r':[0.05,0.10,0.20,0.30],'birdnet':[0.392,0.514,0.606,0.639],'routing':[0.447,0.562,0.661,0.693],'ext':[0.385,0.503,0.638,0.684]},
}

fig,axes=plt.subplots(1,3,figsize=(12.5,4.2),sharey=True)
for ax,region,fname,label in [(axes[0],'ID','fig_curve_ID.npz','Indonesia'),(axes[1],'PH','fig_curve_PH.npz','Philippines'),(axes[2],'MY','fig_curve_MY.npz','Malaysia')]:
    z=np.load(D+fname)
    (wb,cb),(ws,cs),(wh,ch)=curves(z['s1'],z['bn_ok'],z['ptop'],z['eo'])
    ax.plot(wb,cb,color='#888888',lw=1.5,label='BirdNET alone')
    ax.plot(ws,cs,color='#1f77b4',lw=1.5,label='extension alone')
    ax.plot(wh,ch,color='#d62728',lw=1.8,label='BirdNET + routing')
    t=table1[region]
    ax.scatter(t['r'],t['birdnet'],marker='o',s=22,color='#888888',zorder=5)
    ax.scatter(t['r'],t['ext'],marker='s',s=22,color='#1f77b4',zorder=5)
    ax.scatter(t['r'],t['routing'],marker='^',s=26,color='#d62728',zorder=5)

    zp=np.load(D+f'perch2_cache_{region}.npz')
    ths_s1=np.quantile(zp['s1'],np.linspace(0,1,101))
    (wbp,cbp),(wsp,csp),(whp,chp)=curves(zp['s1'],zp['bn_ok'],zp['ptop'],zp['eo'],ths_base=ths_s1)
    ax.plot(wbp,cbp,color='#888888',lw=1.3,ls='--',label='Perch 2.0 alone')
    ax.plot(whp,chp,color='#d62728',lw=1.5,ls='--',label='Perch 2.0 + routing')

    ax.set_xlim(0,0.35); ax.set_ylim(0,0.75)
    ax.set_xticks(np.arange(0,0.301,0.05))  # no label at the panel edge: 0.35 collided with the next panel's 0.00
    ax.set_xlabel(r'wrong$_{all}$'); ax.set_title(label)
    ax.grid(alpha=0.25)
axes[0].set_ylabel(r'correct$_{all}$')
axes[-1].legend(loc='lower right',fontsize=7,frameon=False)
fig.tight_layout(w_pad=1.5)
fig.savefig(__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/../fig_hybrid_curves.pdf')
print('saved fig_hybrid_curves.pdf')
