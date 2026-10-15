import pickle,json,itertools,numpy as np,pandas as pd
from scipy.stats import spearmanr
from data import build,FEATURES
# Choi & Lee (2019) Table 3 (per 100 g slag) and Table 4; NaOH given as dry mass; l/b = 0.42 incl. silicate water
T=[('4-0.75',4,.75,3.89,10.60,280,355),('4-1.00',4,1.0,3.47,14.13,65,112),('4-1.25',4,1.25,3.04,17.67,33,53),
   ('6-0.75',6,.75,5.83,15.90,46,60),('6-1.00',6,1.0,5.20,21.20,34,45),('6-1.25',6,1.25,4.56,26.50,24,40),
   ('8-0.75',8,.75,7.78,21.20,19,28),('8-1.00',8,1.0,6.93,28.27,17,27),('8-1.25',8,1.25,6.08,35.34,16,25)]
rows=[]
for mix,na,ms,naoh,sil,ist,fst in T:
    W=42.0; wS=0.624*sil; extra=W-wS; sio2=ms*na
    rows.append(dict(mix=mix,Na2O_in=na,IST=ist,FST=fst,Na2O_pct=na/100,MS=ms,LS=W/(100+na+sio2),Alk_Bi=(naoh+sil+extra)/100,
                     has_Na2SiO3=1,water_binder_ratio=extra/100,Na2SiO3_binder_ratio=sil/100))
d=pd.DataFrame(rows)
for t in ['IST','FST']: d[t+'_pred']=pickle.load(open(f'H_excl16_{t}.pkl','rb')).predict(d[FEATURES].values)
u=json.load(open('unc.json'))['AD']; mu,sd=np.array(u['mu']),np.array(u['sd'])
v=build(); tr=v[v.valid&(v.trial!=16)]; Z=(tr[FEATURES].values-mu)/sd
d['AD_dist']=[float(np.sort(np.sqrt((((r-mu)/sd-Z)**2).sum(1)))[:3].mean()) for r in d[FEATURES].values]
rng={f:(tr[f].min(),tr[f].max()) for f in FEATURES}
d['n_out']=[sum(not(rng[f][0]-1e-9<=r[f]<=rng[f][1]+1e-9) for f in FEATURES) for _,r in d.iterrows()]
old=pd.read_csv('ext_choi.csv')
print(pd.DataFrame({'mix':d.mix,'wb_old':old.water_binder_ratio.round(3),'wb_new':d.water_binder_ratio.round(3),'AlkBi_old':old.Alk_Bi.round(3),'AlkBi_new':d.Alk_Bi.round(3),
 'IST':d.IST,'IST_pred':d.IST_pred.round(1),'FST':d.FST,'FST_pred_old':old.FST_pred.round(1),'FST_pred_new':d.FST_pred.round(1),'AD':d.AD_dist.round(2),'n_out':d.n_out}).to_string(index=False))
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
s=d[d.Na2O_in>=6].reset_index(drop=True); out={'thr':u['threshold']}
for t in ['IST','FST']:
    y=s[t].values.astype(float); p=s[t+'_pred'].values; ly,lp=np.log(y),np.log(np.clip(p,1,None))
    r=dict(spearman=float(spearmanr(y,p)[0]),raw_rmse=rmse(y,p),mean_bias=float(np.mean(p-y)))
    for k in [1,2,3]:
        R={'offset':[],'naive':[]}
        for cal in itertools.combinations(range(len(y)),k):
            cal=list(cal); te=[i for i in range(len(y)) if i not in cal]
            R['offset'].append(rmse(y[te],np.exp(lp[te]+np.mean(ly[cal]-lp[cal])))); R['naive'].append(rmse(y[te],np.full(len(te),np.exp(ly[cal].mean()))))
        for m,vv in R.items(): r[f'k{k}_{m}']=float(np.median(vv))
    out[t]=r
old=json.load(open('ext.json'))
for t in ['IST','FST']:
    print(t,{k:(round(old[t][k],2),round(out[t][k],2)) for k in out[t]})
d.to_csv('ext_choi_v2.csv',index=False); json.dump(out,open('ext_v2.json','w'))
