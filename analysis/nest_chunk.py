import json, sys, os, numpy as np
from model import *; from data import *
scen,tgt,a,b=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
d=build(); v=d[d.valid].reset_index(drop=True)
df=v if scen=='all19' else v[v.trial!=16].reset_index(drop=True)
X=df[FEATURES].values; y=df[tgt].values.astype(float)
fn=f'nest_{scen}_{tgt}.json'; R=json.load(open(fn)) if os.path.exists(fn) else {}
for i in range(a,min(b,len(y))):
    if str(i) in R: continue
    tr=np.arange(len(y))!=i
    p,_,_=pso(X[tr],y[tr],n=8,it=8,seed=SEED+i)
    H=Hybrid(p).fit(X[tr],y[tr])
    R[str(i)]={'trial':int(df.trial[i]),'y':float(y[i]),'pred':float(H.predict(X[i:i+1])[0]),'base':H.base_pred(X[i:i+1])[0].tolist()}
    json.dump(R,open(fn,'w'))
print(len(R),'done')
