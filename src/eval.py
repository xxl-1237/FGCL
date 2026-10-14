import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn import metrics


def _evaluate_once(embeddings, y, ratio=0.1, val_ratio=0.1, random_state=None):

    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    X = normalize(X, norm='l2')

    test_size = 1.0 - ratio - val_ratio  
    
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
   
    val_size_in_rest = val_ratio / (ratio + val_ratio)  
    second_seed = None if random_state is None else random_state + 1
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size_in_rest, random_state=second_seed
    )

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(
        estimator=OneVsRestClassifier(logreg),
        param_grid=dict(estimator__C=c),
        n_jobs=8, cv=5, verbose=0,
    )
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    val_acc = metrics.accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    best_C = clf.best_params_['estimator__C']
    final_clf = OneVsRestClassifier(
        LogisticRegression(solver='liblinear', C=best_C)
    )
    final_clf.fit(
        np.vstack([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )
    y_test_pred = final_clf.predict(X_test)
    test_acc = metrics.accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    return {
        'val_ACC': float(val_acc),
        'val_F1Ma': float(val_f1),
        'test_ACC': float(test_acc),
        'test_F1Ma': float(test_f1),
    }


def label_classification(embeddings, y, ratio=0.1, val_ratio=0.1,
                         n_repeats=50, random_state=None):
 
    results = []
    for _ in range(n_repeats):
        results.append(_evaluate_once(embeddings, y, ratio, val_ratio,
                                       random_state=random_state))

    statistics = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        statistics[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }
    return statistics