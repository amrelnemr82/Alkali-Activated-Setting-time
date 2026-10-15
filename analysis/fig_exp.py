import numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('figs2.py').read().split('d=build()')[0].split('import matplotlib.pyplot as plt, shap')[1])
fig,ax=plt.subplots(1,3,figsize=(7.4,2.7),gridspec_kw={'width_ratios':[1,0.8,1.25]})
# (a) NaOH only: trials 3,4,1 (Alk/Bi 0.55) and 7,6,5 (0.65)
M=[8,10,12]; d={'0.55':([33,24,14],[59,52,50]),'0.65':([50,46,44],[76,77,92])}
a=ax[0]
for k,(ist,fst) in d.items():
    ls='-' if k=='0.55' else '--'
    a.plot(M,ist,ls,marker='o',color=B,ms=4,lw=1.3,label=f'IST, Alk/Bi = {k}')
    a.plot(M,fst,ls,marker='s',color=O,ms=4,lw=1.3,label=f'FST, Alk/Bi = {k}')
a.set_xticks(M); a.set_xlabel('NaOH molarity (M)'); a.set_ylabel('Setting time (min)'); a.set_ylim(0,128); a.set_title('(a) Route A: NaOH only',fontsize=7.5)
a.legend(fontsize=5.4,frameon=False,loc='upper left',ncol=2)
# (b) Ms effect, trials 14,12,13
a=ax[1]; ms=['0.5\n(T14)','1.0\n(T12)','1.3\n(T13)']; x=np.arange(3); w=0.36
a.bar(x-w/2,[18,11,7],w*0.95,color=B,label='IST'); a.bar(x+w/2,[45,23,14],w*0.95,color=O,label='FST')
for i,(p,q) in enumerate(zip([18,11,7],[45,23,14])): a.text(i-w/2,p+1,str(p),ha='center',fontsize=5.8); a.text(i+w/2,q+1,str(q),ha='center',fontsize=5.8)
a.set_xticks(x); a.set_xticklabels(ms,fontsize=6.3); a.set_xlabel('Silicate modulus Ms'); a.set_ylabel('Setting time (min)'); a.set_ylim(0,55); a.set_title('(b) Route B: Alk/Bi = 1.05',fontsize=7.5); a.legend(fontsize=6,frameon=False)
# (c) route C trials
a=ax[2]; T=['T18','T21','T20','T19','T16']; lab=['8.0 %\n0.224','7.0 %\n0.224','6.5 %\n0.216','6.0 %\n0.283','6.0 %\n0.325']
I=[15,70,41,72,135]; Fv=[121,150,146,156,335]; x=np.arange(5)
a.bar(x-w/2,I,w*0.95,color=B,label='IST'); a.bar(x+w/2,Fv,w*0.95,color=O,label='FST')
for i,(p,q) in enumerate(zip(I,Fv)): a.text(i-w/2,p+5,str(p),ha='center',fontsize=5.8); a.text(i+w/2,q+5,str(q),ha='center',fontsize=5.8)
a.set_xticks(x); a.set_xticklabels([f'{t}\n{l}' for t,l in zip(T,lab)],fontsize=5.8); a.set_ylabel('Setting time (min)'); a.set_ylim(0,370); a.set_xlabel('Trial, Na₂O % and w/b',fontsize=7)
a.set_title('(c) Route C: Na₂O % and w/b',fontsize=7.5); a.legend(fontsize=6,frameon=False,loc='upper left')
plt.tight_layout(); plt.savefig(F+'fig_exp_trends.png',bbox_inches='tight'); print('ok')
