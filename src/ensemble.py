"""Your diagram's 6 base models + 4 ensembles. Selection on validation only."""
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def build_models(seed=42):
    return {
        "lr":  LogisticRegression(max_iter=2000),
        "xgb": XGBClassifier(n_estimators=300, learning_rate=0.05, eval_metric="logloss", random_state=seed),
        "lgb": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1),
        "cat": CatBoostClassifier(iterations=300, learning_rate=0.05, verbose=0, random_seed=seed),
        "rf":  RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
    }

def _base_probs(art, X):
    Z = art["pre"].transform(X[art["columns"]])
    return {n: m.predict_proba(Z)[:, 1] for n, m in art["models"].items()}

def _combine(art, P, kind):
    names, top3 = art["names"], art["top3"]
    if kind in names: return P[kind]
    if kind == "mean_all":  return np.mean([P[n] for n in names], axis=0)
    if kind == "mean_top3": return np.mean([P[n] for n in top3], axis=0)
    if kind == "stack_all": return art["meta_all"].predict_proba(np.column_stack([P[n] for n in names]))[:, 1]
    if kind == "stack_top3":return art["meta_top3"].predict_proba(np.column_stack([P[n] for n in top3]))[:, 1]
    raise ValueError(kind)

def predict_ensemble(art, X, kind=None):
    return _combine(art, _base_probs(art, X), kind or art["best"])

def train_ensemble(Xtr, ytr, Xva, yva, seed=42):
    pre = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]).fit(Xtr)  # train only
    Ztr, Zva = pre.transform(Xtr), pre.transform(Xva)
    models = build_models(seed); names = list(models); val_p = {}
    for n, m in models.items():
        m.fit(Ztr, ytr); val_p[n] = m.predict_proba(Zva)[:, 1]
    top3 = sorted(names, key=lambda n: -roc_auc_score(yva, val_p[n]))[:3]
    # stacking meta-learner on OUT-OF-FOLD train predictions (no leakage)
    oof = {n: cross_val_predict(clone(models[n]), Ztr, ytr, cv=5, method="predict_proba")[:, 1] for n in names}
    meta_all  = LogisticRegression(max_iter=1000).fit(np.column_stack([oof[n] for n in names]), ytr)
    meta_top3 = LogisticRegression(max_iter=1000).fit(np.column_stack([oof[n] for n in top3]), ytr)
    art = dict(pre=pre, models=models, names=names, top3=top3, meta_all=meta_all, meta_top3=meta_top3,
               columns=list(Xtr.columns), best=None)
    kinds = names + ["mean_all", "mean_top3", "stack_all", "stack_top3"]
    scores = {k: roc_auc_score(yva, _combine(art, val_p, k)) for k in kinds}
    art["best"] = max(scores, key=scores.get); art["val_scores"] = scores
    return art
