"""Patient-level repeated stratified split generation with explicit overlap checks."""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

def repeated_patient_splits(patient_ids, targets, n_splits=5, n_repeats=10, random_state=0):
    patient_ids=np.asarray(patient_ids); targets=np.asarray(targets)
    if len(np.unique(patient_ids)) != len(patient_ids): raise ValueError("Patient IDs must be unique.")
    cv=RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=random_state)
    for fold,(train,test) in enumerate(cv.split(np.zeros(len(targets)),targets)):
        if set(patient_ids[train]) & set(patient_ids[test]): raise AssertionError("Patient overlap.")
        yield fold,train,test
