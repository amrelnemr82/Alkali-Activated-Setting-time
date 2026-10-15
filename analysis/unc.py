import json,numpy as np
from data import build,FEATURES
out={}
# --- conformal-style prediction intervals from nested-LOOCV residuals (log scale)
for s in ['excl16_FST','excl16_IST','all19_FST','all19_IST']:
    n=json.load(open(f'nest_{s}.json')); y=np.array([v['y'] for v in n.values()]); p=np.array([v['pred'] for v in n.values()])
    p=np.clip(p,1,None); r=np.abs(np.log(y)-np.log(p)); N=len(r); res={}
    for a in (0.2,0.1):
        k=int(np.ceil((1-a)*(N+1))); q=np.sort(r)[min(k,N)-1]
        cov=np.mean([abs(r[i])<=np.sort(np.delete(r,i))[min(int(np.ceil((1-a)*N)),N-1)-1] for i in range(N)])
        res[f'{int((1-a)*100)}']={'q_log':float(q),'factor':float(np.exp(q)),'loo_coverage':float(cov)}
    out[s]=res; print(s,{k:(round(v['factor'],2),round(v['loo_coverage'],2)) for k,v in res.items()})
# --- applicability domain: mean distance to k=3 nearest training mixes in standardized space
v=build(); val=v[v.valid].reset_index(drop=True); tr=val[val.trial!=16].reset_index(drop=True)
X=tr[FEATURES].values; mu,sd=X.mean(0),X.std(0); Z=(X-mu)/sd
def knn(z,Zref,k=3): d=np.sqrt(((Zref-z)**2).sum(1)); return float(np.sort(d)[:k].mean())
dtr=np.array([knn(Z[i],np.delete(Z,i,0)) for i in range(len(Z))]); q1,q3=np.percentile(dtr,[25,75]); thr=float(q3+1.5*(q3-q1))
z16=(val[val.trial==16][FEATURES].values[0]-mu)/sd
out['AD']={'train_d':dtr.tolist(),'trials':tr.trial.tolist(),'threshold':thr,'trial16_d':knn(z16,Z),'mu':mu.tolist(),'sd':sd.tolist()}
print('AD threshold',round(thr,3),'train d range',round(dtr.min(),2),round(dtr.max(),2),'T16',round(out['AD']['trial16_d'],3))
json.dump(out,open('unc.json','w'),indent=1)
