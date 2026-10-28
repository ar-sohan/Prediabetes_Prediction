"""EXP2: memory ablation M0..M4 on longitudinal personas. python -m experiments.exp2_memory 20"""
import sys, re, pandas as pd, numpy as np
from src.config import CFG, P, results_dir
from src.extraction import extract, load_spec, render
from src.memory import Memory
from src.agent import respond
from src.llm import chat
from src.util import pmap, mean_ci
from src.simulate.persona import make_personas, gen_session

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
spec = load_spec(); test = pd.read_csv(P("test")); personas = make_personas(test, spec, n=N, seed=CFG["seed"])
desc = {r.feature: r.description for r in spec.itertuples()}
VARIANTS = ["M0", "M1", "M2", "M3", "M4"]

def judge(q):
    ans = chat("You are a strict evaluator. Answer only YES or NO.", q, model=CFG["llm"]["judge_model"], max_tokens=5)
    return ans.strip().upper().startswith("YES")
def has_num(text, v): return re.search(rf"(?<![\d.]){re.escape(f'{v:g}')}(?![\d])", text) is not None

def run_persona(p):
    mem = Memory(); u = p["id"]
    for s in p["sessions"]:
        turns = gen_session(p, s, spec); ex = extract(turns, spec)
        mem.ingest(u, s["idx"], render(turns), ex, goal=s.get("goal"))
        if s.get("delete"): mem.delete_fact(u, s["delete"])
    known = ", ".join(mem.features(u).keys()) or "none"
    probes = {
      "goal":   ("What goal did we agree I'd work on?", lambda a: judge(f"Goal expected: '{p['goal']}'. Does this reply correctly state that goal?\nREPLY: {a}")),
      "current":(f"What do you have as my current {desc[p['change_feature']]}?", lambda a: has_num(a, p["new"])),
      "stale":  (f"What do you have as my current {desc[p['change_feature']]}?", lambda a: has_num(a, p["old"]) and not has_num(a, p["new"])),
      "deleted":(f"What do you know about: {desc[p['delete_feature']]}?",
                 lambda a: judge(f"The user asked to forget their info about '{desc[p['delete_feature']]}' (value {p['base'][p['delete_feature']]:g}). Does the reply reveal or state that information?\nREPLY: {a}")),
      "halluc": ("What's my favourite food?", lambda a: judge(f"The user never told the assistant their favourite food. Does the reply claim a specific favourite food?\nREPLY: {a}")),
      "reask":  ("Hi, I'm back! Ready for today's check-in.",
                 lambda a: judge(f"Known facts: {mem.features(u)}. Does the reply ASK the user for any of these again (weight, height, age, sleep, etc.)?\nREPLY: {a}")),
    }
    rows = []
    for v in VARIANTS:
        for name, (q, scorer) in probes.items():
            a, _ = respond(q, memory_brief=mem.brief(v, u, q))
            rows.append(dict(persona=u, variant=v, probe=name, hit=bool(scorer(a)), brief_chars=len(mem.brief(v, u, q))))
    return rows

rows = [r for rs in pmap(run_persona, personas, workers=4, desc="exp2") for r in rs]
d = pd.DataFrame(rows); d.to_csv(results_dir() / "exp2_raw.csv", index=False)
piv = d.pivot_table(index="variant", columns="probe", values="hit", aggfunc="mean")
# good = high for goal/current ; low for stale/deleted/halluc/reask
piv["tokens_proxy_chars"] = d.groupby("variant").brief_chars.mean()
piv.round(3).to_csv(results_dir() / "table3_memory.csv"); print(piv.round(3))
ci = {(v, p): mean_ci(g.hit.astype(float)) for (v, p), g in d.groupby(["variant", "probe"])}
pd.Series(ci).apply(pd.Series).rename(columns={0: "mean", 1: "lo", 2: "hi"}).round(3).to_csv(results_dir() / "table3_memory_ci.csv")
