"""Fold-safe primitive for PTMv2 model evaluation."""
from sklearn.linear_model import LogisticRegression
from preprocessing import make_preprocessing_pipeline

def fit_score_fold(X_train,y_train,X_test,threshold,C,l1_ratio):
    prep=make_preprocessing_pipeline(threshold)
    Xt=prep.fit_transform(X_train); Xv=prep.transform(X_test)
    model=LogisticRegression(penalty='elasticnet',solver='saga',C=C,l1_ratio=l1_ratio,max_iter=10000,random_state=0)
    return model.fit(Xt,y_train).predict_proba(Xv)[:,1]


def fit_score_fold_pca(X_train, y_train, X_test, threshold, n_components, C, l1_ratio):
    """PCA-reduce on the training fold, then fit low-dim saga logistic."""
    from sklearn.decomposition import PCA

    prep = make_preprocessing_pipeline(threshold)
    Xt = prep.fit_transform(X_train)
    Xv = prep.transform(X_test)
    pca = PCA(n_components=n_components, random_state=0)
    Xt_low = pca.fit_transform(Xt)
    Xv_low = pca.transform(Xv)
    model = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio, max_iter=10000, random_state=0)
    return model.fit(Xt_low, y_train).predict_proba(Xv_low)[:, 1]
