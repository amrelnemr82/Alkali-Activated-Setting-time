import json, pickle, sys, time, numpy as np
from model import *; from data import *
scen,tgt=sys.argv[1],sys.argv[2]
d=build(); v=d[d.valid].reset_index(drop=True)
df=v if scen=='all19' else v[v.trial!=16].reset_index(drop=True)
X=df[FEATURES].values; y=df[tgt].values.astype(float)
t=time.time(); p,f,h=pso(X,y); H=Hybrid(p).fit(X,y)
out={'pso_params':[float(a) for a in p],'pso_cv_rmse':float(f),'pso_hist':[float(a) for a in h],'trials':df.trial.tolist(),'y':y.tolist()}
for j,k in enumerate(['RF','GB','ANN']): out[k]={'pred':H.oof[:,j].tolist(),**metrics(y,H.oof[:,j])}
hp=H.meta.predict(H.oof)
out['Hybrid']={'pred':hp.tolist(),**metrics(y,hp),'meta_coef':H.meta.coef_.tolist(),'meta_int':float(H.meta.intercept_)}
out['train_fit']=metrics(y,H.predict(X))
json.dump(out,open(f'res_{scen}_{tgt}.json','w'),indent=1); pickle.dump(H,open(f'H_{scen}_{tgt}.pkl','wb'))
print(scen,tgt,np.round(p,2),round(f,2),{k:(round(out[k]['R2'],3),round(out[k]['RMSE'],1)) for k in ['RF','GB','ANN','Hybrid']},round(time.time()-t))
