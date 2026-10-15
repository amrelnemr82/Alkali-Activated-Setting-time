import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('figs2.py').read().split('d=build()')[0].split('import matplotlib.pyplot as plt, shap')[1])
S=json.load(open('summary.json')); out={}
fig,ax=plt.subplots(1,2,figsize=(6.4,2.5))
for a,t,lab in zip(ax,['FST','IST'],'ab'):
    v=np.array(list(json.load(open(f'yrand_{t}.json')).values())); r=S[f'excl16_{t}']['Hybrid']['R2']
    p=(1+(v>=r).sum())/(len(v)+1); out[t]=dict(n=len(v),mean=float(v.mean()),sd=float(v.std()),max=float(v.max()),true=r,p=p,n_ge=int((v>=r).sum()))
    a.hist(v,bins=np.linspace(-0.2,1,25),color='#9fbfe6',edgecolor='white',lw=0.6,label=f'Permuted responses (n = {len(v)})')
    a.axvline(r,color=O,lw=1.6,label=f'True responses, R² = {r:.2f}')
    a.set_xlabel('Standard-LOOCV R² of the hybrid'); a.set_ylabel('Count'); a.set_title(f'({lab}) {t}: empirical p = {p:.3f}',fontsize=7.5)
    a.set_ylim(0,13); a.legend(fontsize=5.8,frameon=False,loc='upper left')
plt.tight_layout(); plt.savefig(F+'fig_yrand.png',bbox_inches='tight'); print(out); json.dump(out,open('yrand_summary.json','w'))
