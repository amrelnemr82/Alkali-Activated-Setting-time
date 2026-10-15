import numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, skew
exec(open('figs2.py').read().split('d=build()')[0].split('import matplotlib.pyplot as plt, shap')[1])
v=build(); v=v[v.valid].reset_index(drop=True)
cols=FEATURES+['IST','FST']; fig,ax=plt.subplots(3,3,figsize=(7,6.6))
XL={'Na2O_pct':'Na$_2$O / slag (–)','MS':'SiO$_2$/Na$_2$O (–)','LS':'Water / solids (–)','Alk_Bi':'Alkaline solution / slag (–)','has_Na2SiO3':'Silicate present (0 = no, 1 = yes)','water_binder_ratio':'Extra water / slag (–)','Na2SiO3_binder_ratio':'Na$_2$SiO$_3$ solution / slag (–)','IST':'Initial setting time (min)','FST':'Final setting time (min)'}
for a,c in zip(ax.flat,cols):
    x=v[c].values.astype(float); a.hist(x,bins=8,color=B,edgecolor='white',linewidth=1.5,density=True)
    if x.std()>0 and c!='has_Na2SiO3':
        g=np.linspace(x.min()-0.1*np.ptp(x),x.max()+0.1*np.ptp(x),200); a.plot(g,gaussian_kde(x)(g),color=INK,lw=1.6)
    a.set_title(f"{LAB.get(c,c+' (min)')}\nskew = {skew(x,bias=False):.2f}",fontsize=7.5); a.set_yticks([]); a.tick_params(labelsize=6.5); a.set_xlabel(XL[c],fontsize=7)
    if list(ax.flat).index(a)%3==0: a.set_ylabel('Density',fontsize=7)
plt.tight_layout(); plt.savefig(F+'fig_dist_kde.png',bbox_inches='tight'); print('ok')
