import numpy as np, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
SEED=42
def rf(): return RandomForestRegressor(n_estimators=300,max_depth=4,random_state=SEED)
def gb(): return GradientBoostingRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,random_state=SEED)
def ann(p):
    h,la,ll=p
    return MLPRegressor(hidden_layer_sizes=(int(round(h)),),alpha=10**la,learning_rate_init=10**ll,
                        solver='lbfgs',max_iter=3000,random_state=SEED)
# PSO bounds: hidden neurons, log10 alpha, log10 lr (lr used if adam; kept for fidelity of 3-D search)
LB=np.array([2,-4,-3.5]); UB=np.array([32,1,-1.5])
def cv_rmse(p,X,y,k=5):
    kf=KFold(k,shuffle=True,random_state=SEED); e=[]
    for tr,te in kf.split(X):
        sc=StandardScaler().fit(X[tr]); ys=y[tr].std()+1e-9; ym=y[tr].mean()
        m=ann(p).fit(sc.transform(X[tr]),(y[tr]-ym)/ys)
        e.append(np.mean((m.predict(sc.transform(X[te]))*ys+ym-y[te])**2))
    return np.sqrt(np.mean(e))
def pso(X,y,n=15,it=15,seed=SEED):
    r=np.random.default_rng(seed); d=3
    x=r.uniform(LB,UB,(n,d)); v=np.zeros((n,d))
    pb=x.copy(); pf=np.array([cv_rmse(p,X,y) for p in x]); g=pb[pf.argmin()].copy(); gf=pf.min(); hist=[gf]
    for t in range(it-1):
        w=0.9-0.5*t/(it-1)
        v=w*v+1.5*r.random((n,d))*(pb-x)+1.5*r.random((n,d))*(g-x)
        x=np.clip(x+v,LB,UB)
        f=np.array([cv_rmse(p,X,y) for p in x])
        m=f<pf; pb[m]=x[m]; pf[m]=f[m]
        if pf.min()<gf: gf=pf.min(); g=pb[pf.argmin()].copy()
        hist.append(gf)
    return g,gf,hist
class ScaledANN:
    def __init__(s,p): s.p=p
    def fit(s,X,y): s.ym=y.mean(); s.ys=y.std()+1e-9; s.m=ann(s.p).fit(X,(y-s.ym)/s.ys); return s
    def predict(s,X): return s.m.predict(X)*s.ys+s.ym
def base_models(p): return {'RF':rf(),'GB':gb(),'ANN':ScaledANN(p)}
class Hybrid:
    def __init__(s,p,alpha=1.0): s.p=p; s.alpha=alpha
    def fit(s,X,y):
        s.sc=StandardScaler().fit(X); Xs=s.sc.transform(X)
        oof=np.zeros((len(y),3))
        for tr,te in LeaveOneOut().split(Xs):
            for j,(k,m) in enumerate(base_models(s.p).items()):
                oof[te,j]=m.fit(Xs[tr],y[tr]).predict(Xs[te])
        s.oof=oof
        s.meta=Ridge(alpha=s.alpha).fit(oof,y)
        s.base={k:m.fit(Xs,y) for k,m in base_models(s.p).items()}
        return s
    def base_pred(s,X):
        Xs=s.sc.transform(X); return np.column_stack([s.base[k].predict(Xs) for k in ['RF','GB','ANN']])
    def predict(s,X): return s.meta.predict(s.base_pred(X))
def metrics(y,p):
    return dict(R2=r2_score(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),MAE=mean_absolute_error(y,p),
                MAPE=np.mean(np.abs((y-p)/y))*100)
