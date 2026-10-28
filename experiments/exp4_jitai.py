"""EXP4: JITAI policies in simulation + sensitivity analysis. python -m experiments.exp4_jitai"""
import pandas as pd
from src.config import results_dir
from src.simulate.user_sim import run_policy
from src.util import mean_ci
rows = []
for seed in range(5):
    for pol in ["none", "random", "fixed", "tailored"]:
        rows.append(dict(seed=seed, **run_policy(pol, seed=seed)))
d = pd.DataFrame(rows); d.to_csv(results_dir() / "exp4_raw.csv", index=False)
t4 = d.groupby("policy").agg(["mean", "std"]).drop(columns="seed").round(4); t4.to_csv(results_dir() / "table4_jitai.csv"); print(t4)
sens = []                                          # sensitivity: invented parameters varied
for eff in [0.04, 0.08, 0.12]:
    for bias in [-2.0, -1.0, 0.0]:
        for pol in ["none", "random", "fixed", "tailored"]:
            for seed in range(3): sens.append(dict(effect=eff, engage_bias=bias, seed=seed, **run_policy(pol, seed=seed, effect=eff, engage_bias=bias)))
s = pd.DataFrame(sens).groupby(["effect", "engage_bias", "policy"])[["health_gain", "msgs_per_week", "high_risk_escalated"]].mean().round(4)
s.to_csv(results_dir() / "exp4_sensitivity.csv"); print(s)
