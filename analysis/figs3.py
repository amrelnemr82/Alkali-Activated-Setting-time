import json,pickle,warnings,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from data import build,FEATURES
exec(open('figs2.py').read().split('d=build()')[0].split('import matplotlib.pyplot as plt, shap')[1])
def save(n): plt.savefig(F+n,bbox_inches='tight'); plt.close(); print('saved',n)
U=json.load(open('unc.json'))
# ---------- Prediction intervals + applicability domain
fig,ax=plt.subplots(1,3,figsize=(7.3,2.6))
for a,t,lab in zip(ax[:2],['FST','IST'],'ab'):
    n=json.load(open(f'nest_excl16_{t}.json')); y=np.array([q['y'] for q in n.values()]); p=np.clip(np.array([q['pred'] for q in n.values()]),1,None)
    f=U[f'excl16_{t}']['80']['factor']; lo,hi=p/f,p*f; inside=(y>=lo)&(y<=hi)
    a.errorbar(y,p,yerr=[p-lo,hi-p],fmt='none',ecolor='#9fbfe6',elinewidth=1.2,capsize=2)
    a.scatter(y[inside],p[inside],s=16,color=B,edgecolor='white',lw=0.4,zorder=3,label='Measured inside 80 % interval')
    a.scatter(y[~inside],p[~inside],s=20,marker='D',color=O,edgecolor=INK,lw=0.4,zorder=3,label='Outside')
    m=max(y.max(),hi.max())*1.05; a.plot([0,m],[0,m],color=MUT,lw=0.7,ls='--'); a.set_xlim(0,m); a.set_ylim(0,m)
    a.set_xlabel(f'Measured {t} (min)'); a.set_ylabel(f'Nested prediction (min)'); a.set_title(f'({lab}) {t}, 80 % interval:\nfrom prediction / {f:.2f} to prediction × {f:.2f}',fontsize=6.8)
    if t=='FST': a.legend(fontsize=5.8,frameon=False,loc='upper left')
a=ax[2]; dtr=np.array(U['AD']['train_d']); thr=U['AD']['threshold']
n=json.load(open('nest_excl16_FST.json')); y=np.array([q['y'] for q in n.values()]); p=np.array([q['pred'] for q in n.values()]); sr=(p-y)/np.std(p-y)
a.scatter(dtr,sr,s=16,color=B,edgecolor='white',lw=0.4,label='Training mixes (LOO)')
e=pd.read_csv('ext_choi.csv'); a.scatter(e.AD_dist,np.clip((e.FST_pred-e.FST)/np.std(p-y),-6,6),marker='s',s=16,color=Y,edgecolor=INK,lw=0.4,label='Choi & Lee mixes (clipped)')
a.scatter([U['AD']['trial16_d']],[np.clip((147-335)/np.std(p-y),-6,6)],marker='*',s=60,color=O,edgecolor=INK,lw=0.4,label='Trial 16 (held out)')
a.axvline(thr,color=MUT,ls='--',lw=0.8); a.axhline(3,color='#e6e5e0',lw=0.8); a.axhline(-3,color='#e6e5e0',lw=0.8)
a.text(thr,-6.6,' AD limit',fontsize=6,color=MUT); a.set_xlabel('Mean distance to 3 nearest training mixes'); a.set_ylabel('Standardised FST residual'); a.set_title('(c) Applicability domain (FST)',fontsize=7.5)
a.legend(fontsize=5.5,frameon=False,loc='lower left',bbox_to_anchor=(0.0,0.13)); a.set_ylim(-7,7.2)
plt.tight_layout(); save('fig_pi_ad.png')
# ---------- External transferability
e=pd.read_csv('ext_choi.csv'); E=json.load(open('ext.json'))
fig,ax=plt.subplots(1,2,figsize=(6.4,2.8))
for a,t,lab in zip(ax,['IST','FST'],'ab'):
    a.scatter(e[t],e[t+'_pred'],marker='o',facecolors='none',edgecolors=MUT,s=26,label='Raw prediction (9 mixes)')
    s=e[e.Na2O_in>=6]; ly,lp=np.log(s[t].values),np.log(s[t+'_pred'].values)
    pc=[np.exp(lp[i]+np.median([ly[j]-lp[j] for j in range(len(s)) if j!=i])) for i in range(len(s))]
    a.scatter(s[t],pc,s=24,color=B,edgecolor='white',lw=0.4,label='After one-mix recalibration')
    hi=max(e[t].max(),e[t+'_pred'].max())*1.15; a.plot([5,hi],[5,hi],color=MUT,lw=0.7,ls='--'); a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(8,hi); a.set_ylim(8,hi)
    a.set_xlabel(f'Measured {t} (min)'); a.set_ylabel(f'Predicted {t} (min)')
    a.set_title(f"({lab}) {t}: ρ = {E[t]['spearman']:.2f}, raw RMSE {E[t]['raw_rmse']:.0f} → {E[t]['k1_offset']:.0f} min",fontsize=7.3)
ax[0].legend(fontsize=6,frameon=False,loc='upper left'); plt.tight_layout(); save('fig_external.png')
# ---------- Design charts (FST), one per activator route
HF=pickle.load(open('H_excl16_FST.pkl','rb')); v=build(); tr=v[v.valid&(v.trial!=16)]
R={'A':[1,3,4,5,6,7],'B':list(range(8,16)),'C':[18,19,20,21]}
spec={'A':('Na2O_pct','Alk_Bi'),'B':('Na2O_pct','MS'),'C':('water_binder_ratio','Na2O_pct')}
fig,ax=plt.subplots(1,3,figsize=(7.3,2.6))
lv=[20,30,45,60,90,120,150,180]
for a,(r,(fx,fy)),lab in zip(ax,spec.items(),'abc'):
    s=tr[tr.trial.isin(R[r])]; base=s[FEATURES].median().values
    px=0.07*(s[fx].max()-s[fx].min()); py=0.07*(s[fy].max()-s[fy].min()); gx=np.linspace(s[fx].min()-px,s[fx].max()+px,60); gy=np.linspace(s[fy].min()-py,s[fy].max()+py,60); GX,GY=np.meshgrid(gx,gy)
    Z=np.tile(base,(GX.size,1)); Z[:,FEATURES.index(fx)]=GX.ravel(); Z[:,FEATURES.index(fy)]=GY.ravel()
    P=np.clip(HF.predict(Z).reshape(GX.shape),0,None)
    cf=a.contourf(GX,GY,P,levels=12,cmap=SEQ); cs=a.contour(GX,GY,P,levels=[l for l in lv if P.min()<l<P.max()],colors='white',linewidths=0.6); a.clabel(cs,fmt='%d',fontsize=5.5)
    a.scatter(s[fx],s[fy],s=18,color=O,edgecolor='white',lw=0.5,zorder=3)
    for _,q in s.iterrows(): a.annotate(f"T{int(q.trial)}",(q[fx],q[fy]),xytext=(2,2),textcoords='offset points',fontsize=5,color=INK)
    a.set_xlabel(LAB[fx],fontsize=7); a.set_ylabel(LAB[fy],fontsize=7); a.set_title(f'({lab}) Route {r}: predicted FST (min)',fontsize=7.3); a.tick_params(labelsize=6)
    cb=fig.colorbar(cf,ax=a,fraction=0.046,pad=0.02); cb.ax.tick_params(labelsize=5.5)
plt.tight_layout(); save('fig_design.png')
# ---------- Validation schematic
fig,ax=plt.subplots(figsize=(7.2,2.9)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,42)
def box(x,y,w,h,txt,fc='white',ec=INK,fs=5.9,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='square,pad=0',fc=fc,ec=ec,lw=0.7)); ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,weight='bold' if bold else 'normal',linespacing=1.3)
def arr(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=0.7,color=INK,mutation_scale=7))
box(0,4,33,34,'',fc='#f4f4f4',ec='#9a9a9a'); ax.text(16.5,35,'(a) Standard LOOCV',ha='center',fontsize=7.5,weight='bold')
box(2,24,29,8,'PSO tunes ANN on all n mixes\n(5-fold CV RMSE)'); arr(16.5,24,16.5,21)
box(2,12,29,9,'Internal LOOCV → out-of-fold\npredictions of RF, GB, ANN'); arr(16.5,12,16.5,10)
box(2,5,29,5,'Ridge fitted and scored on OOF')
box(35,4,65,34,'',fc='#f4f4f4',ec='#9a9a9a'); ax.text(67,35,'(b) Nested LOOCV (honest estimate)',ha='center',fontsize=7.5,weight='bold')
box(38,25,20,7,'Outer loop i = 1…n\nwithhold mix i'); arr(58,28.5,62,28.5)
box(62,13,34,19,'',fc='white',ec=B); ax.text(79,29.5,'Inner pipeline on n − 1 mixes',ha='center',fontsize=6.6,color=B)
box(65,22.5,28,5,'PSO tunes ANN (8 × 8)'); arr(79,22.5,79,21)
box(65,16,28,5,'Internal LOOCV stacking + refit'); arr(79,16,79,12)
box(62,6,34,6,'Predict withheld mix i (never seen)'); arr(62,9,58,9)
box(38,6,20,6,'Pool n predictions\n→ R², RMSE, MAE')
save('fig_validation_scheme.png')
