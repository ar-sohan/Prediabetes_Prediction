"""STEP 1: leakage screen -> train FULL and CHAT-FEASIBLE models -> calibrate -> test once.
Run:  python -m src.step1_train
Needs data/train.csv, val.csv, test.csv (numeric features + target) and data/feature_spec.csv.
"""
import json, joblib, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score, brier_score_loss,
                             f1_score, recall_score, precision_score)
from .config import CFG, P, models_dir, results_dir
from .ensemble import train_ensemble, predict_ensemble

def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1); e = 0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def boot_auc(y, p, n=1000, seed=0):
    r = np.random.default_rng(seed); y = np.asarray(y); a = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if len(set(y[i])) == 2: a.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

class Platt:
    def fit(self, p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        self.lr = LogisticRegression().fit(np.log(p / (1 - p)).reshape(-1, 1), y); return self
    def transform(self, p):
        p = np.clip(p, 1e-6, 1 - 1e-6); return self.lr.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]

def main():
    T, seed = CFG["target"], CFG["seed"]
    tr, va, te = [pd.read_csv(P(k)) for k in ("train", "val", "test")]
    spec = pd.read_csv(P("feature_spec"))
    all_cols = [c for c in tr.columns if c != T]
    bad = [c for c in all_cols if not pd.api.types.is_numeric_dtype(tr[c])]
    assert not bad, f"Encode these non-numeric columns first: {bad}"
    chat_cols = [c for c in spec[spec.chat_class.isin(["easy", "noisy"])].feature if c in all_cols]

    rows = []
    for c in all_cols:                                   # leakage screen: single-feature AUC
        a = roc_auc_score(tr[T], tr[c].fillna(tr[c].median())); rows.append((c, max(a, 1 - a)))
    leak = pd.DataFrame(rows, columns=["feature", "single_feature_auc"]).sort_values("single_feature_auc", ascending=False)
    leak.to_csv(results_dir() / "leakage_screen.csv", index=False); print(leak.head(8))

    summary = {}
    for name, cols in {"full": all_cols, "chat": chat_cols}.items():
        art = train_ensemble(tr[cols], tr[T], va[cols], va[T], seed)
        pv, pt = predict_ensemble(art, va[cols]), predict_ensemble(art, te[cols])
        calib = (Platt() if CFG["model"]["calibration"] == "platt" else IsotonicRegression(out_of_bounds="clip")).fit(pv, va[T])
        pv_c, pt_c = calib.transform(pv), calib.transform(pt)
        fpr, tpr, thr = roc_curve(va[T], pv_c)                       # thresholds from VALIDATION only
        youden = float(thr[np.argmax(tpr - fpr)])
        high = float(max(thr[np.where(fpr <= 0.10)[0].max()], youden))
        tiers = {"low_max": youden, "high_min": high}
        yhat = (pt_c >= youden).astype(int)                          # TEST evaluated ONCE
        lo, hi = boot_auc(te[T], pt_c)
        summary[name] = dict(n_features=len(cols), best=art["best"], val_auc=art["val_scores"][art["best"]],
                             test_AUC=roc_auc_score(te[T], pt_c), AUC_CI_low=lo, AUC_CI_high=hi,
                             PR_AUC=average_precision_score(te[T], pt_c), Brier=brier_score_loss(te[T], pt_c),
                             ECE=ece(te[T].values, pt_c), Recall=recall_score(te[T], yhat),
                             Precision=precision_score(te[T], yhat), F1=f1_score(te[T], yhat))
        joblib.dump(dict(art=art, calib=calib), models_dir() / f"pipeline_{name}.joblib")
        json.dump(dict(features=cols, tiers=tiers, best=art["best"], val_scores=art["val_scores"],
                       model_version=f"{name}_v1", seed=seed),
                  open(models_dir() / f"meta_{name}.json", "w"), indent=2)
        if name == "chat":
            top = art["models"][art["top3"][0]] if art["top3"][0] in art["models"] else art["models"]["lgb"]
            r = permutation_importance(top, art["pre"].transform(va[cols]), va[T],
                                       scoring="roc_auc", n_repeats=10, random_state=seed)
            pd.DataFrame({"feature": cols, "importance": r.importances_mean}).sort_values(
                "importance", ascending=False).to_csv(results_dir() / "feature_importance_chat.csv", index=False)
    tbl = pd.DataFrame(summary); tbl.to_csv(results_dir() / "table1_model_performance.csv"); print(tbl.round(4))

if __name__ == "__main__":
    main()
