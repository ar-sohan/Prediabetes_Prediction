"""EXP1: extraction accuracy + effect on risk score (true vs extracted features). python -m experiments.exp1_extraction 60"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.config import CFG, P, results_dir
from src.extraction import extract, load_spec
from src.risk import RiskModel
from src.util import pmap
from src.simulate.conversation_gen import generate, CONDITIONS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
T = CFG["target"]; spec = load_spec(); rm = RiskModel("chat")
test = pd.read_csv(P("test")); sub = test.sample(N, random_state=CFG["seed"]).reset_index(drop=True)
jobs = [(i, c) for i in range(N) for c in CONDITIONS]

def run(job):
    i, cond = job; truth = {f: float(sub.loc[i, f]) for f in spec.feature if f in sub and sub.loc[i, f] == sub.loc[i, f]}
    turns, revealed = generate(truth, spec, cond, seed=i)
    return dict(i=i, cond=cond, revealed=revealed, truth=truth, ext=extract(turns, spec))
res = pmap(run, jobs, workers=8, desc="exp1")

rows = []
for r in res:
    for f in spec.feature:
        if f not in r["truth"]: continue
        v = r["ext"][f]["value"]; rev = f in r["revealed"]; t = r["truth"][f]
        rows.append(dict(cond=r["cond"], feature=f, revealed=rev, predicted=v is not None, hallucinated=(v is not None and not rev),
                         abs_err=(abs(v - t) if (v is not None and rev) else np.nan),
                         within10=(abs(v - t) <= 0.1 * max(abs(t), 1) if (v is not None and rev) else np.nan)))
d = pd.DataFrame(rows)
tp = d[d.revealed & d.predicted].groupby(["cond", "feature"]).size()
acc = d.groupby(["cond", "feature"]).apply(lambda g: pd.Series(dict(
    recall=(g.revealed & g.predicted).sum() / max(g.revealed.sum(), 1),
    precision=(g.revealed & g.predicted).sum() / max(g.predicted.sum(), 1),
    hallucination_rate=g.hallucinated.sum() / max((~g.revealed).sum(), 1),
    MAE=g.abs_err.mean(), within10=g.within10.mean())), include_groups=False)
acc.round(3).to_csv(results_dir() / "table2_extraction.csv"); print(acc.round(3))

# risk score: true features vs extracted features
cols = rm.meta["features"]; out = []
for cond in CONDITIONS:
    rr = [r for r in res if r["cond"] == cond]
    Xt = pd.DataFrame([{c: sub.loc[r["i"], c] if c in sub else np.nan for c in cols} for r in rr])
    Xe = pd.DataFrame([{c: r["ext"][c]["value"] if c in r["ext"] else np.nan for c in cols} for r in rr])
    y = sub.loc[[r["i"] for r in rr], T].values
    pt, pe = rm.score_batch(Xt), rm.score_batch(Xe)
    out.append(dict(cond=cond, AUC_true=roc_auc_score(y, pt), AUC_extracted=roc_auc_score(y, pe),
                    mean_abs_prob_diff=float(np.mean(np.abs(pt - pe))), corr=float(np.corrcoef(pt, pe)[0, 1])))
pd.DataFrame(out).round(3).to_csv(results_dir() / "exp1_risk_true_vs_extracted.csv", index=False); print(pd.DataFrame(out).round(3))
