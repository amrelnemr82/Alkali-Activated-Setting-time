import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
from data import build, FEATURES
d=build(); v=d[d.valid].reset_index(drop=True)
cols=FEATURES+['IST','FST']
lab={'Na2O_pct':'Na$_2$O','MS':'Ms','LS':'L/S','Alk_Bi':'Alk/Bi','has_Na2SiO3':'has_Na$_2$SiO$_3$',
     'water_binder_ratio':'w/b','Na2SiO3_binder_ratio':'Na$_2$SiO$_3$/b','IST':'IST','FST':'FST'}
C=v[cols].astype(float).corr().values
fig,ax=plt.subplots(figsize=(9,7.5))
im=ax.imshow(C,cmap='RdBu_r',vmin=-1,vmax=1)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j,i,f'{C[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(C[i,j])>0.6 else 'black')
L=[lab[c] for c in cols]
ax.set_xticks(range(len(cols))); ax.set_xticklabels(L,rotation=45,ha='right'); ax.set_yticks(range(len(cols))); ax.set_yticklabels(L)
cb=fig.colorbar(im,ax=ax); cb.set_label('Pearson r')
ax.set_title('Correlation Heatmap -- Mix Design Features & Setting Time',fontsize=14)
plt.tight_layout(); plt.savefig(sys.argv[1] if len(sys.argv)>1 else 'figures/fig_corr.png',dpi=600); print(np.round(C[-1],2))
