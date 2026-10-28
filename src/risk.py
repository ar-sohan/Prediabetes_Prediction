"""Background risk scoring from chat-derived features (frozen model)."""
import json, joblib, numpy as np, pandas as pd
from .config import CFG, models_dir
from .ensemble import predict_ensemble

class RiskModel:
    def __init__(self, which="chat"):
        d = joblib.load(models_dir() / f"pipeline_{which}.joblib")
        self.art, self.calib = d["art"], d["calib"]
        self.meta = json.load(open(models_dir() / f"meta_{which}.json"))
        self.cols, self.tiers = self.meta["features"], self.meta["tiers"]
        self.min_feats = CFG["model"]["min_features_for_score"]

    def derive(self, f):
        f = dict(f)
        if "bmi" in self.cols and f.get("bmi") is None and f.get("weight_kg") and f.get("height_cm"):
            f["bmi"] = f["weight_kg"] / (f["height_cm"] / 100) ** 2   # computed in code, never by the LLM
        return f

    def score(self, features: dict):
        f = self.derive(features)
        row = pd.DataFrame([{c: (np.nan if f.get(c) is None else float(f[c])) for c in self.cols}])
        n = int(row.notna().sum(axis=1).iloc[0])
        if n < self.min_feats:
            return dict(probability=None, tier="insufficient_data", n_features=n, model_version=self.meta["model_version"])
        p = float(self.calib.transform(predict_ensemble(self.art, row))[0])
        tier = "high" if p >= self.tiers["high_min"] else ("moderate" if p >= self.tiers["low_max"] else "low")
        return dict(probability=p, tier=tier, n_features=n, model_version=self.meta["model_version"])

    def score_batch(self, df):
        """df: rows with chat feature columns (NaN allowed). Returns calibrated probabilities."""
        return self.calib.transform(predict_ensemble(self.art, df[self.cols]))
