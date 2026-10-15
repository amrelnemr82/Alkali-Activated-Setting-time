import json,sys,os,time,numpy as np,warnings; warnings.filterwarnings('ignore')
from model import Hybrid,metrics; from data import build,FEATURES
t=sys.argv[1]; a,b=int(sys.argv[2]),int(sys.argv[3])
v=build(); v=v[v.valid&(v.trial!=16)].reset_index(drop=True); X=v[FEATURES].values; y=v[t].values.astype(float)
p=json.load(open(f'res_excl16_{t}.json'))['pso_params']
fn=f'yrand_{t}.json'; R=json.load(open(fn)) if os.path.exists(fn) else {}
for k in range(a,b):
    if str(k) in R: continue
    ys=np.random.default_rng(1000+k).permutation(y); t0=time.time()
    H=Hybrid(p).fit(X,ys); R[str(k)]=metrics(ys,H.meta.predict(H.oof))['R2']; json.dump(R,open(fn,'w'))
print(t,len(R),round(time.time()-t0,1))
