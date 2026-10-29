"""EXP3b: how few features can the model use? Trains the full ensemble on the top-k chat features.
Ranking uses TRAIN-only LightGBM importance (no validation/test peeking). k is chosen on VALIDATION.
Run:  python -m experiments.exp3b_core_features            (default k = 5,10,15,20,30)
      python -m experiments.exp3b_core_features 5,10,20     (custom list)
Needs Step 1 outputs (models/pipeline_chat.joblib, meta_chat.json). Takes about 3 min per k."""
import sys, json, joblib, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from src.config import CFG, P, models_dir, results_dir
from src.ensemble import train_ensemble, predict_ensemble
from src.step1_train import boot_auc, ece, Platt

T, seed = CFG["target"], CFG["seed"]
tr, va, te = [pd.read_csv(P(k)) for k in ("train", "val", "test")]
meta = json.load(open(models_dir() / "meta_chat.json")); cols = meta["features"]
art0 = joblib.load(models_dir() / "pipeline_chat.joblib")["art"]
rank = pd.Series(art0["models"]["lgb"].feature_importances_, index=cols).sort_values(ascending=False)
rank.to_csv(results_dir() / "core_feature_ranking.csv", header=["lgb_importance"]); print(rank.head(25))

ks = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [5, 10, 15, 20, 30]
ks = [k for k in ks if k < len(cols)]; rows, saved = [], {}
for k in ks:
    c = list(rank.index[:k])
    art = train_ensemble(tr[c], tr[T], va[c], va[T], seed)
    pv, pt = predict_ensemble(art, va[c]), predict_ensemble(art, te[c])
    calib = (Platt() if CFG["model"]["calibration"] == "platt" else IsotonicRegression(out_of_bounds="clip")).fit(pv, va[T])
    pv_c, pt_c = calib.transform(pv), calib.transform(pt)
    fpr, tpr, thr = roc_curve(va[T], pv_c); youden = float(thr[np.argmax(tpr - fpr)])
    tiers = dict(low_max=youden, high_min=float(max(thr[np.where(fpr <= 0.10)[0].max()], youden)))
    lo, hi = boot_auc(te[T], pt_c)
    rows.append(dict(k=k, best=art["best"], val_auc=art["val_scores"][art["best"]], test_AUC=roc_auc_score(te[T], pt_c),
                     CI_low=lo, CI_high=hi, Brier=brier_score_loss(te[T], pt_c), ECE=ece(te[T].values, pt_c)))
    saved[k] = (dict(art=art, calib=calib), dict(features=c, tiers=tiers, best=art["best"], val_scores=art["val_scores"],
                                                  model_version=f"core{k}_v1", seed=seed))
    print(pd.DataFrame(rows).round(4).to_string(index=False))          # progress is visible as it runs
d = pd.DataFrame(rows); d.round(4).to_csv(results_dir() / "exp3b_core_features.csv", index=False)

# choose the SMALLEST k whose VALIDATION AUC is within 0.01 of the best (validation only)
ref = max(d.val_auc.max(), meta["val_scores"][meta["best"]]); ok = d[d.val_auc >= ref - 0.01]
kstar = int(ok.k.min()) if len(ok) else int(d.loc[d.val_auc.idxmax(), "k"])
joblib.dump(saved[kstar][0], models_dir() / "pipeline_core.joblib"); json.dump(saved[kstar][1], open(models_dir() / "meta_core.json", "w"), indent=2)
print(f"\nCHOSEN core size k={kstar} (smallest within 0.01 val-AUC of best). Features:\n{saved[kstar][1]['features']}")
