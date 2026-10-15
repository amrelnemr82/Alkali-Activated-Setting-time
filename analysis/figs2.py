import json,pickle,warnings,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt, shap
from scipy.stats import gaussian_kde, skew
from data import build,FEATURES
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'figure.dpi':300,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','grid.color':'#e6e5e0'})
B,O,A,Y='#2a78d6','#eb6834','#1baf7a','#eda100'; INK='#0b0b0b'; MUT='#52514e'
SEQ=matplotlib.colors.LinearSegmentedColormap.from_list('seq',['#cfe0f5','#2a78d6','#0b2f5e'])
F='figures/'
LAB={'Na2O_pct':'Na$_2$O (fraction of slag)','MS':'Ms (SiO$_2$/Na$_2$O)','LS':'L/S','Alk_Bi':'Alk/Bi','has_Na2SiO3':'has_Na$_2$SiO$_3$','water_binder_ratio':'w/b (extra water/slag)','Na2SiO3_binder_ratio':'Na$_2$SiO$_3$/b'}
SH={'Na2O_pct':'Na$_2$O','MS':'Ms','LS':'L/S','Alk_Bi':'Alk/Bi','has_Na2SiO3':'Silicate flag','water_binder_ratio':'w/b','Na2SiO3_binder_ratio':'Na$_2$SiO$_3$/b'}
d=build(); v=d[d.valid].reset_index(drop=True); v18=v[v.trial!=16].reset_index(drop=True)
def save(n): plt.savefig(F+n,bbox_inches='tight'); plt.close(); print('saved',n)
# ---------- Fig: distributions with KDE (19 mixes)
cols=FEATURES+['IST','FST']; fig,ax=plt.subplots(3,3,figsize=(7,6))
for a,c in zip(ax.flat,cols):
    x=v[c].values.astype(float); a.hist(x,bins=8,color='#cfe0f5',edgecolor='white',linewidth=1.5,density=True)
    if x.std()>0 and c!='has_Na2SiO3':
        g=np.linspace(x.min()-0.1*np.ptp(x),x.max()+0.1*np.ptp(x),200); a.plot(g,gaussian_kde(x)(g),color=B,lw=1.6)
    a.set_title(f"{LAB.get(c,c+' (min)')}\nskew = {skew(x):.2f}",fontsize=7.5); a.set_yticks([]); a.tick_params(labelsize=6.5)
plt.tight_layout(); save('fig_dist_kde.png')
# ---------- Taylor diagrams
def taylor(ax,y,preds,title):
    sr=np.std(y); ax.set_thetamin(0); ax.set_thetamax(90)
    rmax=1.6*sr; ax.set_rlim(0,rmax)
    for r in np.arange(0.5,1.7,0.5)*sr: ax.plot(np.linspace(0,np.pi/2,100),[r]*100,color='#e6e5e0',lw=0.6)
    th=np.linspace(0,np.pi/2,100)
    for c in (0.5,1.0,1.5):  # centred RMS arcs
        rr=[];tt=[]
        for t in th:
            # solve r^2 + sr^2 - 2 r sr cos t = (c sr)^2
            b=-2*sr*np.cos(t); cc=sr**2-(c*sr)**2; disc=b*b-4*cc
            if disc>=0:
                r=(-b+np.sqrt(disc))/2
                if 0<=r<=rmax: rr.append(r); tt.append(t)
        ax.plot(tt,rr,color=Y,lw=0.6,ls='--')
    ax.plot([0],[sr],marker='*',ms=10,color=INK); ax.annotate('Measured',(0,sr),xytext=(4,4),textcoords='offset points',fontsize=6.5)
    ticks=[0.2,0.4,0.6,0.8,0.9,0.95,0.99]; ax.set_thetagrids(np.degrees(np.arccos(ticks)),[str(t) for t in ticks],fontsize=6)
    mk=iter(['o','s','^','D','v','P','X','<','>','h','p','*','d','H','8','o','s','^'])
    FAM={'XGBoost':Y,'DT':Y,'GPR':B,'SVR':B,'Ridge':B,'MLR':B,'KNN':B,'ANN (default)':B}
    for nm,p in preds.items():
        r=np.corrcoef(y,p)[0,1]; s=np.std(p); hyb=nm.startswith('Hybrid')
        ax.plot(np.arccos(np.clip(r,0,1)),min(s,1.55*sr),marker=next(mk),ms=7 if hyb else 5,color=O if hyb else FAM.get(nm,A),mec=INK,mew=0.3,ls='none',label=nm)
    ax.set_title(title,fontsize=8.5,pad=12); ax.set_xlabel('Standard deviation (min)',fontsize=7,labelpad=14); ax.tick_params(labelsize=6)
fig=plt.figure(figsize=(7.2,4.3))
for k,t in enumerate(['FST','IST']):
    b=json.load(open(f'bench_excl16_{t}.json')); y=np.array(b['y']); M=b['models']
    n=json.load(open(f'nest_excl16_{t}.json')); nest=np.array([q['pred'] for q in n.values()])
    preds={'Hybrid (nested)':nest,'Hybrid (LOOCV)':np.array(json.load(open(f'res_excl16_{t}.json'))['Hybrid']['pred'])}
    for nm in ['GPR','Extra Trees','SVR','LightGBM','CatBoost','AdaBoost','Ridge','MLR','KNN','ANN (default)','Bagged DT','RF','GB','XGBoost','DT']: preds[nm]=np.array(M[nm]['pred'])
    ax=fig.add_subplot(1,2,k+1,projection='polar'); taylor(ax,y,preds,f'({"ab"[k]}) {t}, 18 mixes')
    if k==1: ax.legend(fontsize=5.6,loc='upper left',bbox_to_anchor=(1.02,1.05),frameon=False,ncol=1)
plt.tight_layout(); save('fig_taylor.png')
# ---------- Conventional (empirical) vs hybrid
fig,ax=plt.subplots(1,3,figsize=(7.2,2.6),gridspec_kw={'width_ratios':[1,1,1.1]})
for k,sc in enumerate(['all19','excl16']):
    b=json.load(open(f'bench_{sc}_FST.json')); y=np.array(b['y']); e=np.array(b['models']['Empirical']['pred'])
    n=json.load(open(f'nest_{sc}_FST.json')); h=np.array([q['pred'] for q in n.values()])
    a=ax[k]; hi=max(y.max(),e.max(),h.max())*1.08; a.plot([0,hi],[0,hi],color=MUT,lw=0.7,ls='--')
    a.scatter(y,e,marker='s',s=18,color=Y,edgecolor=INK,lw=0.4,label='Empirical equation'); a.scatter(y,h,marker='o',s=18,color=B,edgecolor='white',lw=0.4,label='Hybrid (nested)')
    a.set_xlim(0,hi);a.set_ylim(0,hi);a.set_xlabel('Measured FST (min)');a.set_ylabel('Predicted FST (min)'); a.set_title(f"({'ab'[k]}) {'19 mixes' if sc=='all19' else '18 mixes (T16 excl.)'}",fontsize=8)
    if k==0: a.legend(fontsize=6,frameon=False,loc='upper left')
res={}
for sc in ['all19','excl16']:
    b=json.load(open(f'bench_{sc}_FST.json')); y=np.array(b['y']); n=json.load(open(f'nest_{sc}_FST.json'))
    res[f'Hybrid\n{"19" if sc=="all19" else "18"}']=np.array([q['pred'] for q in n.values()])-y; res[f'Empirical\n{"19" if sc=="all19" else "18"}']=np.array(b['models']['Empirical']['pred'])-y
a=ax[2]; keys=['Hybrid\n18','Empirical\n18','Hybrid\n19','Empirical\n19']
bp=a.boxplot([res[k] for k in keys],tick_labels=keys,widths=0.5,patch_artist=True,showmeans=True,meanprops=dict(marker='s',mfc=INK,mec=INK,ms=3),medianprops=dict(color=INK))
for p,c in zip(bp['boxes'],[B,Y,B,Y]): p.set_facecolor(c); p.set_alpha(0.55)
a.axhline(0,color=MUT,lw=0.7,ls='--'); a.set_ylabel('Residual, predicted − measured (min)'); a.set_title('(c) Residual distributions',fontsize=8); a.tick_params(axis='x',labelsize=6.5)
plt.tight_layout(); save('fig_conventional.png')
# ---------- SHAP importance with % share and groups
X=json.load(open('xai.json')); Xm=np.array(X['X'])
GROUPS={'Alkali chemistry':['Na2O_pct','MS','has_Na2SiO3'],'Water content':['water_binder_ratio','LS'],'Activator proportioning':['Alk_Bi','Na2SiO3_binder_ratio']}
GC={'Alkali chemistry':B,'Water content':O,'Activator proportioning':A}
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
shares={}
for a,t in zip(ax,['FST','IST']):
    sv=np.array(X[t]['sv']); imp=np.abs(sv).mean(0); pct=100*imp/imp.sum(); o=np.argsort(imp)
    col=[GC[[g for g,fs in GROUPS.items() if FEATURES[i] in fs][0]] for i in o]
    a.barh(range(len(o)),imp[o],color=col,height=0.62)
    for r,i in enumerate(o): a.text(imp[i]+imp.max()*0.02,r,f'{imp[i]:.1f} min ({pct[i]:.0f} %)',va='center',fontsize=6.3,color=INK)
    a.set_yticks(range(len(o))); a.set_yticklabels([SH[FEATURES[i]] for i in o]); a.set_xlim(0,imp.max()*1.45)
    a.set_xlabel('Mean |SHAP| (min)'); a.set_title(f"({'ab'[ax.tolist().index(a)]}) {t}",fontsize=8.5)
    shares[t]={g:float(sum(pct[FEATURES.index(f)] for f in fs)) for g,fs in GROUPS.items()}
    shares[t+'_feat']={f:float(pct[i]) for i,f in enumerate(FEATURES)}
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=c,label=f"{g}: FST {shares['FST'][g]:.0f} %, IST {shares['IST'][g]:.0f} %") for g,c in GC.items()],fontsize=6,frameon=False,loc='lower center',ncol=3,bbox_to_anchor=(0.5,-0.07))
plt.tight_layout(); save('fig_shap_bar.png'); json.dump(shares,open('shares.json','w'),indent=1)
# ---------- Beeswarm with sequential ramp
fig,ax=plt.subplots(1,2,figsize=(7.2,3.1))
for a,t in zip(ax,['IST','FST']):
    sv=np.array(X[t]['sv']); o=np.argsort(np.abs(sv).mean(0))
    for r,i in enumerate(o):
        col=Xm[:,i]; cn=(col-col.min())/(np.ptp(col)+1e-12); jit=np.random.default_rng(r).uniform(-.18,.18,len(col))
        a.scatter(sv[:,i],r+jit,c=cn,cmap=SEQ,s=16,edgecolor='white',lw=0.3,vmin=0,vmax=1)
    a.axvline(0,color=MUT,lw=0.6); a.set_yticks(range(len(o))); a.set_yticklabels([SH[FEATURES[i]] for i in o]); a.set_xlabel(f'SHAP value for {t} (min)'); a.set_title(f"({'ab'[['IST','FST'].index(t)]}) {t}",fontsize=8.5)
sm=plt.cm.ScalarMappable(cmap=SEQ); cb=fig.colorbar(sm,ax=ax,fraction=0.025,pad=0.02); cb.set_ticks([0,1]); cb.set_ticklabels(['Low','High']); cb.set_label('Feature value',fontsize=7)
save('fig_shap_beeswarm.png')
# ---------- Dependence plots, FST, all 7 features with interaction colouring
sv=np.array(X['FST']['sv']); fig,ax=plt.subplots(2,4,figsize=(7.4,3.9)); inter={}
for k,f in enumerate(FEATURES):
    a=ax.flat[k]; i=FEATURES.index(f); cand=[j for j in shap.approximate_interactions(i,sv,Xm) if j!=i]; j=int(cand[0]); inter[f]=FEATURES[j]
    c=Xm[:,j]; cn=(c-c.min())/(np.ptp(c)+1e-12); xx=Xm[:,i]+(np.random.default_rng(k).uniform(-.03,.03,len(Xm)) if f=='has_Na2SiO3' else 0)
    sc=a.scatter(xx,sv[:,i],c=cn,cmap=SEQ,s=18,edgecolor=INK,lw=0.3,vmin=0,vmax=1); a.axhline(0,color=MUT,lw=0.6,ls='--')
    a.set_xlabel(LAB[f],fontsize=6.8); a.set_ylabel('SHAP (min)',fontsize=6.8); a.set_title(f'({"abcdefg"[k]}) colour: {SH[FEATURES[j]]}',fontsize=6.8); a.tick_params(labelsize=6)
ax.flat[7].axis('off'); cb=fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ),ax=ax.flat[7],fraction=0.5,aspect=12); cb.set_ticks([0,1]); cb.set_ticklabels(['Low','High']); cb.set_label('Interacting feature value',fontsize=7)
plt.tight_layout(); save('fig_shap_dependence.png'); json.dump(inter,open('inter_FST.json','w'))
# ---------- ICE + PDP
for t in ['FST','IST']:
    ice=X[t]['ice']; fig,ax=plt.subplots(2,4,figsize=(7.4,3.9))
    for k,f in enumerate(FEATURES):
        a=ax.flat[k]; g=np.array(ice[f]['grid']); P=np.array(ice[f]['pred'])
        for row in P: a.plot(g,row,color='#9fbfe6',lw=0.6,alpha=0.8)
        a.plot(g,P.mean(0),color=O,lw=2.0)
        lo,hi=Xm[:,k].min(),Xm[:,k].max(); a.plot(Xm[:,k],np.full(len(Xm),a.get_ylim()[0]),'|',color=INK,ms=5)
        a.set_xlabel(LAB[f],fontsize=6.8); a.set_ylabel(f'Predicted {t} (min)',fontsize=6.8); a.set_title(f'({"abcdefg"[k]})',fontsize=7); a.tick_params(labelsize=6)
    ax.flat[7].axis('off'); ax.flat[7].plot([],[],color='#9fbfe6',lw=1,label='ICE (one curve per mix)'); ax.flat[7].plot([],[],color=O,lw=2,label='PDP (average)'); ax.flat[7].plot([],[],'|',color=INK,label='Observed values'); ax.flat[7].legend(loc='center',fontsize=7,frameon=False)
    plt.tight_layout(); save(f'fig_ice_{t}.png')
