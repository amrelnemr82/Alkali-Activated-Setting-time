import pickle,json,itertools,numpy as np,pandas as pd
from scipy.stats import spearmanr
from data import build,FEATURES
def feats(na2o,ms,W,sil=(0.093,0.283,0.624)):
    sio2=ms*na2o; mS=sio2/sil[1]; na_sil=mS*sil[0]; na_naoh=max(na2o-na_sil,0)
    mN=na_naoh/0.48; wN=0.381*mN; wS=sil[2]*mS; extra=W-wN-wS
    return dict(Na2O_pct=na2o/100,MS=ms,LS=W/(100+na2o+sio2),Alk_Bi=(mN+mS+extra)/100,has_Na2SiO3=1,water_binder_ratio=extra/100,Na2SiO3_binder_ratio=mS/100)
mix=[(4,.75,280,355),(4,1,65,112),(4,1.25,33,53),(6,.75,46,60),(6,1,34,45),(6,1.25,24,40),(8,.75,19,28),(8,1,17,27),(8,1.25,16,25)]
d=pd.DataFrame([dict(mix=f'{n}-{m}',Na2O_in=n,IST=i,FST=f,**feats(n,m,42)) for n,m,i,f in mix])
for t in ['IST','FST']: d[t+'_pred']=pickle.load(open(f'H_excl16_{t}.pkl','rb')).predict(d[FEATURES].values)
u=json.load(open('unc.json'))['AD']; mu,sd=np.array(u['mu']),np.array(u['sd'])
v=build(); tr=v[v.valid&(v.trial!=16)]; Z=(tr[FEATURES].values-mu)/sd
d['AD_dist']=[float(np.sort(np.sqrt((((r-mu)/sd-Z)**2).sum(1)))[:3].mean()) for r in d[FEATURES].values]
rng={f:(tr[f].min(),tr[f].max()) for f in FEATURES}
d['n_out']=[sum(not(rng[f][0]-1e-9<=r[f]<=rng[f][1]+1e-9) for f in FEATURES) for _,r in d.iterrows()]
print(d[['mix','IST','FST','IST_pred','FST_pred','n_out','AD_dist']].round(2).to_string(index=False))
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
s=d[d.Na2O_in>=6].reset_index(drop=True); out={'thr':u['threshold']}
for t in ['IST','FST']:
    y=s[t].values.astype(float); p=s[t+'_pred'].values; ly,lp=np.log(y),np.log(p)
    r=dict(spearman=float(spearmanr(y,p)[0]),raw_rmse=rmse(y,p),mean_bias=float(np.mean(p-y)))
    for k in [1,2,3]:
        R={'offset':[],'naive':[]}
        for cal in itertools.combinations(range(len(y)),k):
            cal=list(cal); te=[i for i in range(len(y)) if i not in cal]
            R['offset'].append(rmse(y[te],np.exp(lp[te]+np.mean(ly[cal]-lp[cal])))); R['naive'].append(rmse(y[te],np.full(len(te),np.exp(ly[cal].mean()))))
        for m,vv in R.items(): r[f'k{k}_{m}']=float(np.median(vv))
    out[t]=r
print(json.dumps(out,indent=0)); d.to_csv('ext_choi.csv',index=False); json.dump(out,open('ext.json','w'))
