import numpy as np, pandas as pd
rows = [
# trial, NaOH_g, Na2SiO3_g, H2O_g, Na2O%, MS, LS, conc, ratio, AlkBi, IST, FST, remark
(1,110,0,0,20.5,0,0.23,'12M',None,0.55,14,50,''),
(2,0,110,0,4.5,3.3,0.30,'8.15% Na2O',None,0.55,None,None,'Formation of solid liquids'),
(3,110,0,0,13.6,0,0.32,'8M',None,0.55,33,59,''),
(4,110,0,0,17.1,0,0.27,'10M',None,0.55,24,52,''),
(5,129.9,0,0,24.2,0,0.26,'12M',None,0.65,44,92,''),
(6,129.9,0,0,20.1,0,0.31,'10M',None,0.65,46,77,''),
(7,129.9,0,0,16.1,0,0.37,'8M',None,0.65,50,76,''),
(8,65,65,0,14.7,0.5,0.30,'12M',1,0.65,10,25,''),
(9,65,65,0,10.7,0.5,0.35,'8M',1,0.65,8,20,''),
(10,93.2,46.6,0,19.2,0.3,0.30,'12M',0.5,0.70,17,32,''),
(11,113.3,56.66,0,23.4,0.3,0.35,'12M',0.5,0.85,24,46,''),
(12,59.3,150.7,0,17.2,1.0,0.46,'12M',2.54,1.05,11,23,''),
(13,44.2,165.9,0,15.0,1.3,0.47,'12M',3.75,1.05,7,14,''),
(14,105.27,105.27,0,23.9,0.5,0.43,'12M',1,1.05,18,45,''),
(15,129.03,90.32,0,27.7,0.4,0.43,'12M',0.7,1.10,22,62,''),
(16,17.47,44.36,64.94,6.0,1.0,0.44,'48% Na2O',2.54,0.63,135,335,'High flowability'),
(17,21.5,70.77,27.77,8.0,1.2,0.34,'48% Na2O',3.29,0.60,None,None,'Very stiff consistency'),
(18,23.3,59.16,44.79,8.0,1.0,0.39,'48% Na2O',2.54,0.64,15,121,'Stiff consistency'),
(19,17.51,44.38,56.6,6.0,1.0,0.42,'48% Na2O',2.53,0.59,72,156,'High flowability'),
(20,18.24,52.8,43.16,6.5,1.1,0.37,'48% Na2O',2.89,0.57,41,146,'Fair consistency, High shrinkage'),
(21,20.44,51.55,44.79,7.0,1.0,0.37,'48% Na2O',2.52,0.58,70,150,'Fair consistency'),
]
cols=['trial','NaOH_g','Na2SiO3_g','ExtraH2O_g','Na2O_pct','MS','LS','conc','ratio','Alk_Bi','IST','FST','remark']
raw=pd.DataFrame(rows,columns=cols)
def build():
    d=raw.copy()
    d['activator_route']=np.where(d.conc.str.contains('%'),'pct_na2o_solid','molar_solution')
    d['NaOH_conc_value']=d.conc.str.extract(r'([\d.]+)').astype(float)
    d['ratio']=d.ratio.fillna(0.0)
    d['Na2O_pct']=d.Na2O_pct/100
    d['has_Na2SiO3']=(d.Na2SiO3_g>0).astype(int)
    d['water_binder_ratio']=d.ExtraH2O_g/200
    d['Na2SiO3_binder_ratio']=d.Na2SiO3_g/200
    d['NaOH_binder_ratio']=d.NaOH_g/200
    d['total_solution_g']=d.NaOH_g+d.Na2SiO3_g+d.ExtraH2O_g
    def wc(r):
        t=r.remark.lower()
        if 'solid liquids' in t: return 'failed_reaction'
        if 'very stiff' in t: return 'very_stiff'
        if 'shrinkage' in t: return 'fair_high_shrinkage'
        if 'fair' in t: return 'fair'
        if 'stiff' in t: return 'stiff'
        if 'flowab' in t: return 'high_flowability'
        return 'normal'
    d['workability_class']=d.apply(wc,axis=1)
    d['valid']=d.IST.notna()
    return d
FEATURES=['Na2O_pct','MS','LS','Alk_Bi','has_Na2SiO3','water_binder_ratio','Na2SiO3_binder_ratio']
