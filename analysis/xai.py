import pickle,json,warnings,numpy as np,shap; warnings.filterwarnings('ignore')
from data import build,FEATURES
v=build(); v=v[v.valid&(v.trial!=16)].reset_index(drop=True); X=v[FEATURES].values
out={'X':X.tolist(),'trials':v.trial.tolist()}
for t in ['IST','FST']:
    H=pickle.load(open(f'H_excl16_{t}.pkl','rb'))
    ex=shap.KernelExplainer(H.predict,X); sv=ex.shap_values(X,nsamples=500,silent=True)
    inter=[int(shap.approximate_interactions(i,sv,X)[0]) for i in range(len(FEATURES))]
    # ICE
    ice={}
    for j,f in enumerate(FEATURES):
        grid=np.array([0,1]) if f=='has_Na2SiO3' else np.linspace(X[:,j].min(),X[:,j].max(),25)
        Z=np.repeat(X[None,:,:],len(grid),0).copy(); Z[:,:,j]=grid[:,None]
        P=H.predict(Z.reshape(-1,X.shape[1])).reshape(len(grid),len(X))
        ice[f]={'grid':grid.tolist(),'pred':P.T.tolist()}
    out[t]={'sv':sv.tolist(),'base':float(ex.expected_value),'inter':inter,'ice':ice,'pred':H.predict(X).tolist()}
    imp=np.abs(sv).mean(0); print(t,'base',round(float(ex.expected_value),1),[(f,round(i,1),round(100*i/imp.sum(),1),FEATURES[k]) for f,i,k in sorted(zip(FEATURES,imp,inter),key=lambda z:-z[1])])
    print('  additivity check max |sum shap + base - pred|',round(float(np.max(abs(sv.sum(1)+ex.expected_value-H.predict(X)))),3))
json.dump(out,open('xai.json','w'))
