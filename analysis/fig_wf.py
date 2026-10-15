import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
exec(open('figs2.py').read().split('d=build()')[0].split('import matplotlib.pyplot as plt, shap')[1])
S=json.load(open('summary.json'))
# ---- performance: standard vs nested
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7))
cases=[('all19','IST'),('all19','FST'),('excl16','IST'),('excl16','FST')]; xl=[f'{t}\n({"19" if s=="all19" else "18"} mixes)' for s,t in cases]
ser=[('RF','RF','#c9d9ef'),('GB','GB','#9fbfe6'),('ANN','ANN','#5b93d6'),('Hybrid','Hybrid, standard LOOCV',B),('Hybrid_nested','Hybrid, nested LOOCV',O)]
w=0.16; x=np.arange(4)
for j,(k,l,c) in enumerate(ser):
    for a,m in zip(ax,['R2','RMSE']):
        v=[S[f'{s}_{t}'][k][m] for s,t in cases]
        a.bar(x+(j-2)*w,v,w*0.92,color=c,label=l,edgecolor='white',lw=0.3)
for a,m,lab in zip(ax,['R²','RMSE (min)'],'ab'):
    a.set_xticks(x); a.set_xticklabels(xl,fontsize=6.5); a.set_ylabel(m); a.set_title(f'({lab}) {m.split(" ")[0]}',fontsize=7.5); a.axhline(0,color=INK,lw=0.5)
ax[0].set_ylim(0,1.12); ax[0].legend(fontsize=5.6,frameon=False,ncol=2,loc='upper left')
plt.tight_layout(); plt.savefig(F+'fig_performance.png',bbox_inches='tight'); plt.close()
# ---- workflow
fig,ax=plt.subplots(figsize=(8.8,5.4)); ax.axis('off'); ax.set_xlim(0,101.5); ax.set_ylim(0,70)
cols=[('(a) Data',['Laboratory dataset|21 AAS mixes, 19 with\nmeasured IST and FST','Feature engineering|7 binder-normalised\ndescriptors','Exploratory analysis|KDE, statistics,\ncorrelation, Trial 16']),
('(b) Modelling',['Base learners|RF, GB fixed; ANN\ntuned by PSO','Stacking|LOOCV out-of-fold\npredictions + Ridge','Benchmarks|15 learners + log-linear\nempirical equation']),
('(c) Validation',['Nested LOOCV|tuning and stacking\nrepeated per fold','Statistical checks|Wilcoxon, Taylor,\ny-randomisation','Uncertainty, domain|80/90 % intervals,\nkNN applicability']),
('(d) Explanation',['SHAP|global shares, beeswarm,\ndependence, interactions','ICE and PDP|individual and average\nfeature effects','External check|Choi and Lee (2019)\nmixes, recalibration']),
('(e) Use',['Design charts|FST contours per\nactivator route','Web application|forward prediction,\ninterval, domain flag','Inverse design|route-constrained\nsearch, lab check'])]
cw=18.6; gap=1.75
for i,(h,bx) in enumerate(cols):
    x0=0.5+i*(cw+gap)
    ax.add_patch(FancyBboxPatch((x0,4),cw,62,boxstyle='square,pad=0',fc='#f4f4f4',ec='#9a9a9a',lw=0.7))
    ax.text(x0+cw/2,62.5,h,ha='center',va='center',fontsize=7.6,weight='bold')
    for j,b in enumerate(bx):
        t,s=b.split('|'); y0=44-j*18.5
        ax.add_patch(FancyBboxPatch((x0+1,y0),cw-2,14,boxstyle='square,pad=0',fc='white',ec=INK,lw=0.7))
        ax.text(x0+cw/2,y0+10.3,t,ha='center',va='center',fontsize=6.3,weight='bold')
        ax.text(x0+cw/2,y0+4.8,s,ha='center',va='center',fontsize=5.6,linespacing=1.25)
        if j<2: ax.annotate('',xy=(x0+cw/2,y0-4.2),xytext=(x0+cw/2,y0-0.3),arrowprops=dict(arrowstyle='-|>',lw=0.7,color=INK,mutation_scale=6))
    if i<4: ax.annotate('',xy=(x0+cw+gap-0.1,35),xytext=(x0+cw+0.1,35),arrowprops=dict(arrowstyle='-|>',lw=0.9,color=INK,mutation_scale=7))
ax.plot([0.5+4*(cw+gap)+cw/2]*2,[4,1],color=MUT,ls='--',lw=0.7); ax.plot([0.5+cw/2,0.5+4*(cw+gap)+cw/2],[1,1],color=MUT,ls='--',lw=0.7)
ax.annotate('',xy=(0.5+cw/2,4),xytext=(0.5+cw/2,1),arrowprops=dict(arrowstyle='-|>',lw=0.7,color=MUT,mutation_scale=6))
ax.text(50,1.8,'new laboratory mixes appended and models retrained',ha='center',fontsize=5.8,style='italic',color=MUT)
plt.savefig(F+'fig_workflow.png',bbox_inches='tight'); print('ok')
