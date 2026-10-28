"""EXP3: risk model robustness to missing chat features. python -m experiments.exp3_missingness"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.config import CFG, P, results_dir
from src.risk import RiskModel
from src.step1_train import ece
rm = RiskModel("chat"); te = pd.read_csv(P("test")); y = te[CFG["target"]].values; X = te[rm.cols].astype(float)
rows = []
for frac in [0, .1, .2, .3, .4, .5, .6, .7, .8]:
    for s in range(5):
        r = np.random.default_rng(s); M = X.mask(r.random(X.shape) < frac)
        p = rm.score_batch(M); rows.append(dict(missing=frac, seed=s, AUC=roc_auc_score(y, p), ECE=ece(y, p),
                                                insufficient=float((M.notna().sum(axis=1) < rm.min_feats).mean())))
d = pd.DataFrame(rows).groupby("missing")[["AUC", "ECE", "insufficient"]].agg(["mean", "std"]).round(4); d.to_csv(results_dir() / "exp3_missingness.csv"); print(d)
