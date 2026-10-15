import json,pickle,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('figs2.py').read().split('d=build()')[0].split('import matplotlib.pyplot as plt, shap')[1])
from data import FEATURES
HI=pickle.load(open('H_excl16_IST.pkl','rb')); HF=pickle.load(open('H_excl16_FST.pkl','rb')); D=json.load(open('../domain.json'))
MU,SD,TZ,THR=np.array(D['mu']),np.array(D['sd']),np.array(D['train_z']),D['threshold']
R={"A":dict(Na2O_pct=(.136,.242),MS=(0,0),LS=(.26,.37),Alk_Bi=(.55,.65),has_Na2SiO3=(0,0),water_binder_ratio=(0,0),Na2SiO3_binder_ratio=(0,0)),
"B":dict(Na2O_pct=(.107,.277),MS=(.3,1.3),LS=(.30,.47),Alk_Bi=(.65,1.1),has_Na2SiO3=(1,1),water_binder_ratio=(0,0),Na2SiO3_binder_ratio=(.23,.83)),
"C":dict(Na2O_pct=(.06,.08),MS=(1.0,1.1),LS=(.37,.42),Alk_Bi=(.57,.64),has_Na2SiO3=(1,1),water_binder_ratio=(.216,.283),Na2SiO3_binder_ratio=(.22,.30))}
def ad(C): Z=(C-MU)/SD; d=np.sqrt(((Z[:,None]-TZ[None])**2).sum(-1)); return np.sort(d,1)[:,:3].mean(1)
tI,tF=24,52; rng=np.random.default_rng(0); out={}; fig,ax=plt.subplots(figsize=(3.6,3.0)); col={'A':B,'B':O,'C':A}
for r,b in R.items():
    C=np.column_stack([rng.uniform(*b[f],5000) for f in FEATURES]); pi,pf=HI.predict(C),HF.predict(C); dd=ad(C)
    ok=(pf>pi)&(pi>=1)&(dd<=THR); s=np.abs(pi-tI)/tI+np.abs(pf-tF)/tF; s[~ok]=np.inf; k=int(np.argmin(s))
    out[r]=dict(x={f:round(float(C[k,i]),3) for i,f in enumerate(FEATURES)},IST=float(pi[k]),FST=float(pf[k]),S=float(s[k]),AD=float(dd[k]),n_ok=int(ok.sum()),
       n_window=int((ok&(pi>=20)&(pi<=30)&(pf>=45)&(pf<=60)).sum()))
    ax.scatter(pi[ok],pf[ok],s=3,color=col[r],alpha=0.35,lw=0,label=f'Route {r}')
ax.add_patch(plt.Rectangle((20,45),10,15,fill=False,ec=INK,lw=0.8,ls='--')); ax.plot(tI,tF,'*',ms=10,color=INK,label='Target (24, 52 min)')
ax.set_xlabel('Predicted IST (min)'); ax.set_ylabel('Predicted FST (min)'); from matplotlib.lines import Line2D
H=[Line2D([],[],marker='o',ls='none',color=col[r],ms=4,label=f'Route {r}') for r in 'ABC']+[Line2D([],[],marker='*',ls='none',color=INK,ms=7,label='Target (24, 52 min)'),Line2D([],[],ls='--',color=INK,lw=0.8,label='Window 20–30 / 45–60 min')]
ax.legend(handles=H,fontsize=5.8,frameon=False,loc='upper left')
plt.tight_layout(); plt.savefig(F+'fig_inverse.png',bbox_inches='tight'); json.dump(out,open('inverse.json','w'),indent=1)
for r,v in out.items(): print(r,v)
