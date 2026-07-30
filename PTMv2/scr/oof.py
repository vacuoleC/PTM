"""Strict fold-wise out-of-fold prediction recording."""
import pandas as pd

def initialise_oof(assignments):
    return assignments.loc[assignments.role=='test',['fold','patient_id','target']].assign(score=float('nan'))

def record_fold_scores(oof, fold, patient_ids, scores):
    ids=list(patient_ids)
    mask=oof.fold.eq(fold)
    if set(oof.loc[mask,'patient_id']) != set(ids): raise ValueError('Scores must match exactly the predefined test patients.')
    if oof.loc[mask,'score'].notna().any(): raise ValueError('Fold already recorded.')
    out=oof.copy(); out.loc[mask,'score']=out.loc[mask,'patient_id'].map(dict(zip(ids,scores)))
    return out
