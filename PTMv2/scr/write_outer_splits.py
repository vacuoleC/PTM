"""Persist the frozen repeated outer patient splits."""
import pandas as pd
from pathlib import Path
from splits import repeated_patient_splits

labels=pd.read_csv(Path(__file__).parents[2]/'PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv')
rows=[]
for fold,tr,te in repeated_patient_splits(labels.patient_id,labels.target,5,10,0):
 for role,idx in [('train',tr),('test',te)]:
  rows += [{'fold':fold,'role':role,'patient_id':p,'target':int(t)} for p,t in zip(labels.patient_id.iloc[idx],labels.target.iloc[idx])]
out=Path(__file__).parents[1]/'outputs/tables/e2_2_outer_split_assignments.csv'; out.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out,index=False); print(out,flush=True)
