import json, sys, warnings, numpy as np
warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor as TT
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from model import metrics
from data import build, FEATURES
S=42
def sc(m): return make_pipeline(StandardScaler(), m)
def ty(m): return TT(regressor=sc(m), transformer=StandardScaler())
MODELS={
 'MLR':lambda: sc(LinearRegression()),
 'Ridge':lambda: sc(Ridge(alpha=1.0)),
 'KNN':lambda: sc(KNeighborsRegressor(n_neighbors=3,weights='distance')),
 'SVR':lambda: ty(SVR(C=10,epsilon=0.05)),
 'GPR':lambda: sc(GaussianProcessRegressor(C(1.0)*RBF(np.ones(7))+WhiteKernel(0.1),normalize_y=True,random_state=S,n_restarts_optimizer=3)),
 'DT':lambda: DecisionTreeRegressor(max_depth=4,random_state=S),
 'Bagged DT':lambda: BaggingRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=200,random_state=S),
 'AdaBoost':lambda: AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),n_estimators=200,learning_rate=0.05,random_state=S),
 'Extra Trees':lambda: ExtraTreesRegressor(n_estimators=300,max_depth=4,random_state=S),
 'RF':lambda: RandomForestRegressor(n_estimators=300,max_depth=4,random_state=S),
 'GB':lambda: GradientBoostingRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,random_state=S),
 'XGBoost':lambda: XGBRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=1.0,min_child_weight=1,random_state=S,n_jobs=1),
 'LightGBM':lambda: LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=8,min_child_samples=3,random_state=S,verbose=-1,n_jobs=1),
 'CatBoost':lambda: CatBoostRegressor(iterations=500,depth=4,learning_rate=0.05,random_seed=S,verbose=0,thread_count=1),
 'ANN (default)':lambda: ty(MLPRegressor(hidden_layer_sizes=(10,),solver='lbfgs',max_iter=3000,random_state=S)),
}
scen=sys.argv[1]; tgt=sys.argv[2]
d=build(); v=d[d.valid].reset_index(drop=True); df=v if scen=='all19' else v[v.trial!=16].reset_index(drop=True)
X=df[FEATURES].values; y=df[tgt].values.astype(float)
out={}
for nm,mk in MODELS.items():
    p=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X): p[te]=mk().fit(X[tr],y[tr]).predict(X[te])
    out[nm]={'pred':p.tolist(),**metrics(y,p)}
    print(f'{nm:14s} R2={out[nm]["R2"]:.3f} RMSE={out[nm]["RMSE"]:.1f}',flush=True)
# conventional log-linear regression (empirical design equation)
from sklearn.linear_model import LinearRegression as LR
cols=['Na2O_pct','MS','water_binder_ratio','Alk_Bi']; Xe=df[cols].values; p=np.zeros(len(y))
for tr,te in LeaveOneOut().split(Xe): p[te]=np.exp(LR().fit(Xe[tr],np.log(y[tr])).predict(Xe[te]))
full=LR().fit(Xe,np.log(y)); out['Empirical']={'pred':p.tolist(),**metrics(y,p),'coef':dict(zip(cols,full.coef_.round(4).tolist())),'intercept':float(full.intercept_)}
print('Empirical',round(out['Empirical']['R2'],3),round(out['Empirical']['RMSE'],1),out['Empirical']['coef'],round(full.intercept_,3))
json.dump({'y':y.tolist(),'trials':df.trial.tolist(),'models':out},open(f'bench_{scen}_{tgt}.json','w'))
