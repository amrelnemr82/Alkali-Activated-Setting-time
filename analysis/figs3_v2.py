import json,pickle,warnings,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from data import build,FEATURES
exec(open('figs2.py').read().split('d=build()')[0].split('import matplotlib.pyplot as plt, shap')[1])
def save(n): plt.savefig(F+n,bbox_inches='tight'); plt.close(); print('saved',n)
U=json.load(open('unc.json'))
# ---------- Prediction intervals + applicability domain
fig,ax=plt.subplots(1,3,figsize=(7.3,2.6))
for a,t,lab in zip(ax[:2],['FST','IST'],'ab'):
    n=json.load(open(f'nest_excl16_{t}.json')); y=np.array([q['y'] for q in n.values()]); p=np.clip(np.array([q['pred'] for q in n.values()]),1,None)
    f=U[f'excl16_{t}']['80']['factor']; lo,hi=p/f,p*f; inside=(y>=lo)&(y<=hi)
    a.errorbar(y,p,yerr=[p-lo,hi-p],fmt='none',ecolor='#9fbfe6',elinewidth=1.2,capsize=2)
    a.scatter(y[inside],p[inside],s=16,color=B,edgecolor='white',lw=0.4,zorder=3,label='Measured inside 80 % interval')
    a.scatter(y[~inside],p[~inside],s=20,marker='D',color=O,edgecolor=INK,lw=0.4,zorder=3,label='Outside')
    m=max(y.max(),hi.max())*1.05; a.plot([0,m],[0,m],color=MUT,lw=0.7,ls='--'); a.set_xlim(0,m); a.set_ylim(0,m)
    a.set_xlabel(f'Measured {t} (min)'); a.set_ylabel(f'Nested prediction (min)'); a.set_title(f'({lab}) {t}, 80 % interval:\nfrom prediction / {f:.2f} to prediction × {f:.2f}',fontsize=6.8)
    if t=='FST': a.legend(fontsize=5.8,frameon=False,loc='upper left')
a=ax[2]; dtr=np.array(U['AD']['train_d']); thr=U['AD']['threshold']
n=json.load(open('nest_excl16_FST.json')); y=np.array([q['y'] for q in n.values()]); p=np.array([q['pred'] for q in n.values()]); sr=(p-y)/np.std(p-y)
a.scatter(dtr,sr,s=16,color=B,edgecolor='white',lw=0.4,label='Training mixes (LOO)')
e=pd.read_csv('ext_choi_v2.csv'); a.scatter(e.AD_dist,np.clip((e.FST_pred-e.FST)/np.std(p-y),-6,6),marker='s',s=16,color=Y,edgecolor=INK,lw=0.4,label='Choi & Lee mixes (clipped)')
a.scatter([U['AD']['trial16_d']],[np.clip((147-335)/np.std(p-y),-6,6)],marker='*',s=60,color=O,edgecolor=INK,lw=0.4,label='Trial 16 (held out)')
a.axvline(thr,color=MUT,ls='--',lw=0.8); a.axhline(3,color='#e6e5e0',lw=0.8); a.axhline(-3,color='#e6e5e0',lw=0.8)
a.text(thr,-6.6,' AD limit',fontsize=6,color=MUT); a.set_xlabel('Mean distance to 3 nearest training mixes'); a.set_ylabel('Standardised FST residual'); a.set_title('(c) Applicability domain (FST)',fontsize=7.5)
a.legend(fontsize=5.5,frameon=False,loc='lower left',bbox_to_anchor=(0.0,0.13)); a.set_ylim(-7,7.2)
plt.tight_layout(); save('fig_pi_ad.png')
# ---------- External transferability
e=pd.read_csv('ext_choi_v2.csv'); E=json.load(open('ext_v2.json'))
fig,ax=plt.subplots(1,2,figsize=(6.4,2.8))
for a,t,lab in zip(ax,['IST','FST'],'ab'):
    a.scatter(e[t],e[t+'_pred'],marker='o',facecolors='none',edgecolors=MUT,s=26,label='Raw prediction (9 mixes)')
    s=e[e.Na2O_in>=6]; ly,lp=np.log(s[t].values),np.log(s[t+'_pred'].values)
    pc=[np.exp(lp[i]+np.median([ly[j]-lp[j] for j in range(len(s)) if j!=i])) for i in range(len(s))]
    a.scatter(s[t],pc,s=24,color=B,edgecolor='white',lw=0.4,label='After one-mix recalibration')
    hi=max(e[t].max(),e[t+'_pred'].max())*1.15; a.plot([5,hi],[5,hi],color=MUT,lw=0.7,ls='--'); a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(8,hi); a.set_ylim(8,hi)
    a.set_xlabel(f'Measured {t} (min)'); a.set_ylabel(f'Predicted {t} (min)')
    a.set_title(f"({lab}) {t}: ρ = {E[t]['spearman']:.2f}, raw RMSE {E[t]['raw_rmse']:.0f} → {E[t]['k1_offset']:.0f} min",fontsize=7.3)
ax[0].legend(fontsize=6,frameon=False,loc='upper left'); plt.tight_layout(); save('fig_external.png')
