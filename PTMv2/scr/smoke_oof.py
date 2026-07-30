"""Run a fixed-parameter OOF smoke check over the frozen outer splits."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from evaluate import fit_score_fold
from oof import initialise_oof,record_fold_scores

root=Path(__file__).parents[1]
x=pd.read_pickle(root/'../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz')
y=pd.read_csv(root/'../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv').set_index('patient_id')
a=pd.read_csv(root/'outputs/tables/e2_2_outer_split_assignments.csv')
o=initialise_oof(a)
for fold in sorted(o.fold.unique()):
 tr=a[(a.fold==fold)&(a.role=='train')].patient_id; te=a[(a.fold==fold)&(a.role=='test')].patient_id
 s=fit_score_fold(x.loc[tr],y.loc[tr,'target'],x.loc[te],.1,.1,.5)
 o=record_fold_scores(o,fold,te,s)
 print(f'[E2.2] completed fold {fold+1}/50',flush=True)
out=root/'outputs/tables/e2_2_fixed_parameter_oof_smoke.csv';o.to_csv(out,index=False);print(out,flush=True)
