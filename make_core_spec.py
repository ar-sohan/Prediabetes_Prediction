"""Build data/feature_spec_core.csv = the core-model features, plus a value_labels column YOU fill from the codebook.
Run after exp3b:  python make_core_spec.py      (re-running keeps labels you already typed)
value_labels format:  1=Male;2=Female      (numeric code = meaning, separated by semicolons)"""
import json, pathlib, pandas as pd, yaml
cfg = yaml.safe_load(open("config.yaml")); out = pathlib.Path("data/feature_spec_core.csv")
meta = json.load(open("models/meta_core.json"))
spec = pd.read_csv(cfg["paths"]["feature_spec"]); tr = pd.read_csv(cfg["paths"]["train"])
core = spec[spec.feature.isin(meta["features"])].copy()
core["rank"] = core.feature.map({f: i + 1 for i, f in enumerate(meta["features"])})
core = core.sort_values("rank")
u = {f: sorted(tr[f].dropna().unique()) for f in core.feature}
core["type"] = ["categorical" if len(u[f]) <= 12 else "numeric" for f in core.feature]
core["values_seen"] = [", ".join(f"{x:g}" for x in u[f]) if len(u[f]) <= 12 else f"{tr[f].min():g} to {tr[f].max():g}" for f in core.feature]
old = pd.read_csv(out).set_index("feature").get("value_labels") if out.exists() else None
core["value_labels"] = [(old.get(f, "") if old is not None else "") for f in core.feature]
core["value_labels"] = core["value_labels"].fillna("")
core.to_csv(out, index=False)
todo = core[(core.type == "categorical") & (core.value_labels == "")]
print(f"{len(core)} core features written to {out}")
print(f"\n{len(todo)} categorical features still need value_labels (fill from your codebook):")
print(todo[["feature", "values_seen", "description"]].to_string(index=False))
